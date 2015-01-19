#include "deviceid.h"
#include "common.h"
#include <cassert>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>


__global__ void MatAdd(fp_t A[N][N], fp_t B[N][N], fp_t C[N][N])
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	C[i][j] = A[i][j] + B[i][j];
}

extern "C" void cuda_addm(fp_t A[N][N], fp_t B[N][N], fp_t C[N][N])
{
	// Kernel invocation with one block of N * N * 1 threads
	int numBlocks = 1;
	dim3 threadsPerBlock(N, N);
	MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}

__global__ void VecAdd(fp_t* A, fp_t* B, fp_t* C)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		C[i] = A[i] + B[i];
}

extern "C" void cuda_add(fp_t* h_A, fp_t* h_B, fp_t* h_C)
{
	size_t size = N * sizeof(fp_t);

	// Allocate vectors in device memory
	fp_t* d_A;
	cudaMalloc(&d_A, size);
	fp_t* d_B;
	cudaMalloc(&d_B, size);
	fp_t* d_C;
	cudaMalloc(&d_C, size);

	// Copy vectors from host memory to device memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	// Invoke kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

	// Copy result from device memory to host memory
	// h_C contains the result in host memory
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

__global__ void kernel_multiply(const Matrix matrix, const Matrix vector, Matrix result_matrix)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < matrix.width && j < matrix.width)
		result_matrix.elements[i * matrix.width + j] = matrix.elements[i * matrix.width + j] * vector.elements[j];
}

extern "C" void cuda_multiply(const Matrix d_matrix, const Matrix d_vector, Matrix d_result)
{
	assert(d_matrix.width == d_vector.width);
	assert(d_result.width == d_matrix.height);
	assert(d_matrix.width == d_matrix.height);
	const int size = d_matrix.width;
	// Invoke kernel
	const int NUM_THREADS = 32;	// 32 * 32 = 1024 = max threads per block by cuda docs
	dim3 threadsPerBlock(NUM_THREADS, NUM_THREADS);
	dim3 numBlocks((size + NUM_THREADS - 1) / threadsPerBlock.x, (size + NUM_THREADS - 1) / threadsPerBlock.y);
	//printf("%d %d\n", numBlocks.x, numBlocks.y);
	kernel_multiply<<<numBlocks, threadsPerBlock>>>(d_matrix, d_vector, d_result);
}


__global__ void reduce(const Matrix d_A, const Matrix d_b, const Matrix d_sum_matrix, Matrix d_x, const Matrix d_last_x, const fp_t alpha)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int M = d_A.width;
	if (i < M) {
		fp_t sum = 0;
		for (int j = 0; j < M; j++) {
			if (j != i) {
				sum += d_sum_matrix.elements[i * M + j];
			}
		}
		d_x.elements[i] = (1 - alpha) * d_last_x.elements[i] + alpha * (d_b.elements[i] - sum) / d_A.elements[i * M + i];
	}
}

extern "C" void cuda_reduce(const Matrix d_A, const Matrix d_b, const Matrix d_sum_matrix, Matrix* d_x, Matrix* d_last_x, const fp_t alpha)
{
	const int size = d_A.width;
	// Invoke kernel
	const int THREADS = 16;
	dim3 threadsPerBlock(THREADS);
	dim3 numBlocks((size + THREADS - 1) / threadsPerBlock.x);
	//printf("%d,%d %d,%d\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
	reduce<<<numBlocks, threadsPerBlock>>>(d_A, d_b, d_sum_matrix, *d_x, *d_last_x, alpha);
	fp_t* c = d_x->elements;
	d_x->elements = d_last_x->elements;
	d_last_x->elements = c;
}

/*
__global__ void jor(const Matrix d_A, const Matrix d_b, Matrix d_sum_matrix, Matrix d_x, Matrix d_last_x, const fp_t alpha)
{
	int iterations = 0;
	const fp_t min_diff = 0.0001;
	const int max_iter = 200;
	fp_t x_diff = 2 * min_diff;

	while ((x_diff > min_diff) && (max_iter < 0 || iterations < max_iter)) {
		// multiply
		const int size = d_A.width;
		// Invoke kernel
		const int NUM_THREADS = 16;
		dim3 threadsPerBlock(NUM_THREADS, NUM_THREADS);
		dim3 numBlocks((size + NUM_THREADS - 1) / threadsPerBlock.x, (size + NUM_THREADS - 1) / threadsPerBlock.y);
		//printf("%d %d\n", numBlocks.x, numBlocks.y);
		kernel_multiply<<<numBlocks, threadsPerBlock>>>(d_A, d_last_x, d_sum_matrix);

		// reduce
		dim3 rthreadsPerBlock(256);
		dim3 rnumBlocks((size + 256 - 1) / rthreadsPerBlock.x);
		//printf("%d %d\n", numBlocks.x, threadsPerBlock.x);
		reduce<<<rnumBlocks, rthreadsPerBlock>>>(d_A, d_b, d_sum_matrix, d_x, d_last_x, alpha);
		fp_t* c = d_x.elements;
		d_x.elements = d_last_x.elements;
		d_last_x.elements = c;

		// diff
		x_diff = 0;
		for (int i = 0; i < d_x.size; i++) {
			x_diff += fabs(d_x.elements[i] - d_last_x.elements[i]);
		}

		iterations++;
	}
	
}

extern "C" void cuda_jor(const Matrix d_A, const Matrix d_b, Matrix d_sum_matrix, Matrix d_x, Matrix d_last_x, const fp_t alpha)
{
	jor<<<1, 1>>>(d_A, d_b, d_sum_matrix, d_x, d_last_x, alpha);
}
*/

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
