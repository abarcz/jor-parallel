#include "deviceid.h"
#include "common.h"
#include <cassert>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

texture<float, cudaTextureType1D, cudaReadModeElementType> texRef;

extern "C" void texbind(float* array, int size) {
	texRef.addressMode[0] = cudaAddressModeClamp;
	texRef.filterMode = cudaFilterModePoint;
	texRef.normalized = false;
	size_t offset;
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture(&offset, texRef, (void*)array, desc, size);
}

extern "C" void texunbind() {
	cudaUnbindTexture(texRef);
}

// Thread block size
#define BLOCK_SIZE 128

__global__ void kernel_multiply(const Matrix matrix, const Matrix vector, Matrix result_vector)
{
	const int i = blockIdx.x;
	if (i > matrix.height)
		return;
	const int k = threadIdx.x;
	const int shift = BLOCK_SIZE;
	fp_t res = 0;
	#ifdef TEXTURE
		fp_t Avalue = 0;
	#endif
	for (int j = 0 + k; j < matrix.width; j += shift) {
		#ifdef TEXTURE
			Avalue = tex1Dfetch(texRef, i * matrix.width + j);
			res += Avalue * vector.elements[j];
		#else
			res += matrix.elements[i * matrix.width + j] * vector.elements[j];
		#endif
	}
	__shared__ fp_t sdata[BLOCK_SIZE];
	sdata[k] = res;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (k < s) {
			sdata[k] += sdata[k + s];
		}
		__syncthreads();
	}


	if (k == 0) {
		res = sdata[k];
		res -= matrix.elements[i * matrix.width + i] * vector.elements[i];
		result_vector.elements[i] = res;
	}
}

extern "C" void cuda_multiply(const Matrix d_matrix, const Matrix d_vector, Matrix d_result)
{
	assert(d_matrix.width == d_vector.width);
	assert(d_result.width == d_matrix.height);
	assert(d_matrix.width == d_matrix.height);
	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 numBlocks(d_matrix.height);
	//printf("%d,%d %d,%d\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
	kernel_multiply<<<numBlocks, threadsPerBlock>>>(d_matrix, d_vector, d_result);
}

__global__ void reduce(const Matrix d_A, const Matrix d_b, const Matrix d_c, Matrix d_x, const Matrix d_last_x, const fp_t alpha)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int M = d_A.width;
	if (i < M) {
		d_x.elements[i] = (1 - alpha) * d_last_x.elements[i] + alpha * (d_b.elements[i] - d_c.elements[i]) / d_A.elements[i * M + i];
	}
}

extern "C" void cuda_reduce(const Matrix d_A, const Matrix d_b, const Matrix d_c, Matrix* d_x, Matrix* d_last_x, const fp_t alpha)
{
	const int size = d_A.width;
	// Invoke kernel
	const int THREADS = 16;
	dim3 threadsPerBlock(THREADS);
	dim3 numBlocks((size + THREADS - 1) / threadsPerBlock.x);
	//printf("%d,%d %d,%d\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
	reduce<<<numBlocks, threadsPerBlock>>>(d_A, d_b, d_c, *d_x, *d_last_x, alpha);
	fp_t* c = d_x->elements;
	d_x->elements = d_last_x->elements;
	d_last_x->elements = c;
}

__global__ void diff(const Matrix d_a, const Matrix d_b, fp_t* result)
{
	fp_t max_diff = 0;
	fp_t curr_diff;
	for (int i = 0; i < d_a.size; i++) {
		curr_diff = fabs(d_a.elements[i] - d_b.elements[i]);
		if (curr_diff > max_diff)
			max_diff = curr_diff;
	}
	*result = max_diff;
}


extern "C" void cuda_diff(const Matrix d_a, const Matrix d_b, fp_t* result)
{
	diff<<<1, 1>>>(d_a, d_b, result);
}

extern "C" void cuda_identify()
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }
}
