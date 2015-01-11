#include "deviceid.h"
#include <stdio.h>
#include <cuda_runtime.h>


__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	C[i][j] = A[i][j] + B[i][j];
}

extern "C" void cuda_addm(float A[N][N], float B[N][N], float C[N][N])
{
	// Kernel invocation with one block of N * N * 1 threads
	int numBlocks = 1;
	dim3 threadsPerBlock(N, N);
	MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}

__global__ void VecAdd(float* A, float* B, float* C)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		C[i] = A[i] + B[i];
}

extern "C" void cuda_add(float* h_A, float* h_B, float* h_C)
{
	size_t size = N * sizeof(float);

	// Allocate vectors in device memory
	float* d_A;
	cudaMalloc(&d_A, size);
	float* d_B;
	cudaMalloc(&d_B, size);
	float* d_C;
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

__global__ void kernel_multiply(float* matrix, float* vector, float* result, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
		result[i] = matrix[i] * vector[i];
}

extern "C" void cuda_multiply(float* d_matrix, float* d_vector, float* d_result, int size)
{
	// Invoke kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	kernel_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_vector, d_result, size);
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
