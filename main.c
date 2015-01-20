
#include "amatrix.h"
#include "deviceid.h"
#include "common.h"
#include "test.h"
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 *
 * Jacobi Over Relaxation using OpenMP
 *
 * Author: Aleksy Barcz
 *
 */


void swap(Matrix* a, Matrix* b)
{
	assert(a->height == b->height);
	assert(a->width == b->width);
	fp_t* c = a->elements;
	a->elements = b->elements;
	b->elements = c;
}

const bool verbose = false;

void print_cuda_elapsed(fp_t start_time)
{
	cudaDeviceSynchronize();
	printf("%dms spent\n", (int)((omp_get_wtime() - start_time) * 1000));
}

extern int cuda_identify();
extern void cuda_multiply(Matrix d_matrix, Matrix d_vector, Matrix d_result);
extern void cuda_reduce(const Matrix d_A, const Matrix d_b, const Matrix d_c, Matrix* d_x, Matrix* d_last_x, const fp_t alpha);
extern void cuda_diff(const Matrix d_a, const Matrix d_b, fp_t* result);
extern void texbind(float* array, int size);
extern void texunbind();


Matrix* device_matrix(const int width)
{
	Matrix* m = malloc(sizeof(Matrix));
	m->width = width;
	m->height = width;
	m->size = width * width;
	cudaMalloc((void**)&m->elements, m->size * sizeof(fp_t));
	return m;
}

Matrix* device_matrix_from(const Matrix* host_matrix)
{
	Matrix* m = malloc(sizeof(Matrix));
	m->width = host_matrix->width;
	m->height = host_matrix->height;
	m->size = host_matrix->size;
	cudaMalloc((void**)&m->elements, m->size * sizeof(fp_t));
	return m;
}

/** call: ./main <matrix_dimension> <number_of_tests> <use_gpu>*/
int main(int argc, char* argv[])
{
	cuda_identify();

	if (argc != 4) {
		printf("program must be called with arguments: matrix_dimension tests_number use_gpu(0/1)\n");
		exit(1);
	}
	const int M = atoi(argv[1]);
	printf("Using matrix dimension: %d\n", M);
	const int tests = atoi(argv[2]);
	const bool cpu = !atoi(argv[3]);

	// always use the same seed to get the same matrices during tests
	srand(0);

	#ifdef DOUBLE
		const fp_t min_diff = 0.00000001;	//for double, fails with 8192 and floats on both cpu and gpu
	#else
		const fp_t min_diff = 0.000001;
	#endif
	const fp_t alpha = 0.9;
	const int max_iter = 50;

	fp_t* exec_times = malloc(tests * sizeof(fp_t));
	fp_t* all_rmse = malloc(tests * sizeof(fp_t));
	for (int k = 0; k < tests; k++) {

		const DataSet dataset = generate_dataset(M);

		Matrix* last_x = aligned_vector(M, true);
		Matrix* x = aligned_vector(M, true);
		for (int i = 0; i < M; i++) {
		}

		int iterations = 0;

		// solve Ax = b
		const fp_t start_time = omp_get_wtime();

		fp_t sum = 0;
		int j = 0;
		int i = 0;
		const Matrix* A = dataset.A;
		const Matrix* b = dataset.b;
		assert(x != last_x);

		if (cpu) {
			//#pragma omp parallel shared(last_x, x, iterations) private(i, j, sum)
			while ((matrix_diff(x, last_x) > min_diff) && (max_iter < 0 || iterations < max_iter)) {
				//fp_t st_time0 = omp_get_wtime();
				//#pragma omp single
				{
					swap(last_x, x);
				}

				// A, M, alpha and b are constant, so they cannot be declared as shared
				//#pragma omp for schedule(dynamic)
				for (i = 0; i < M; i++) {
					sum = 0;

					//#pragma omp simd aligned(A, last_x: 16) reduction(+:sum) linear(j)
					for (j = 0; j < M; j++) {
						sum += A->elements[i * M + j] * last_x->elements[j];
					}

					sum -= A->elements[i * M + i] * last_x->elements[i];	// opt: outside the loop for sse optimizer
					x->elements[i] = (1 - alpha) * last_x->elements[i] + alpha * (b->elements[i] - sum) / A->elements[i * M + i];
				}

				//#pragma omp single nowait
				{
					iterations++;
				}
				//printf("%dus spent\n", (int)((omp_get_wtime() - st_time0) * 1000000));
			}
		} else {
			Matrix* d_A = device_matrix_from(A);
			#ifndef DOUBLE
				#ifdef TEXTURE
					texbind(d_A->elements, d_A->size * sizeof(fp_t));
				#endif
			#endif
			cudaMemcpy(d_A->elements, A->elements, A->size * sizeof(fp_t), cudaMemcpyHostToDevice);

			Matrix* d_b = device_matrix_from(b);
			cudaMemcpy(d_b->elements, b->elements, b->size * sizeof(fp_t), cudaMemcpyHostToDevice);

			Matrix* d_last_x = device_matrix_from(last_x);
			Matrix* d_c = device_matrix_from(b);
			Matrix* d_x = device_matrix_from(x);
			cudaMemcpy(d_x->elements, x->elements, x->size * sizeof(fp_t), cudaMemcpyHostToDevice);
			cudaMemcpy(d_last_x->elements, last_x->elements, last_x->size * sizeof(fp_t), cudaMemcpyHostToDevice);

			fp_t x_diff = 2 * min_diff;
			fp_t* d_x_diff;
			cudaMalloc((void**)&d_x_diff, sizeof(fp_t));

			//fp_t stime;
			while ((x_diff > min_diff) && (max_iter < 0 || iterations < max_iter)) {
				//stime = omp_get_wtime();
				cuda_multiply(*d_A, *d_last_x, *d_c);
				//print_cuda_elapsed(stime);

				//stime = omp_get_wtime();
				cuda_reduce(*d_A, *d_b, *d_c, d_x, d_last_x, alpha); //performs swap
				//print_cuda_elapsed(stime);

				//stime = omp_get_wtime();
				cuda_diff(*d_x, *d_last_x, d_x_diff);
				//print_cuda_elapsed(stime);

				iterations++;
				//cudaMemcpyFromSymbol(&x_diff, "d_x_diff", sizeof(x_diff), 0, cudaMemcpyDeviceToHost);
				//stime = omp_get_wtime();
				cudaMemcpy(&x_diff, d_x_diff, sizeof(fp_t), cudaMemcpyDeviceToHost);
				//print_cuda_elapsed(stime);
			}
			// copy last_x instead, as it was swapped
			cudaMemcpy(x->elements, d_last_x->elements, x->size * sizeof(fp_t), cudaMemcpyDeviceToHost);

			#ifndef DOUBLE
				#ifdef TEXTURE
					texunbind();
				#endif
			#endif
			cudaFree(d_A->elements);
			cudaFree(d_b->elements);
			cudaFree(d_last_x->elements);
			cudaFree(d_c->elements);
			cudaFree(d_x->elements);
			cudaFree(d_x_diff);

			free(d_A);
			free(d_b);
			free(d_c);
			free(d_last_x);
			free(d_x);
		}
		const fp_t end_time = omp_get_wtime();
		const fp_t seconds_spent = end_time - start_time;
		exec_times[k] = seconds_spent;

		if (verbose) {
			printf("x: ");
			print_matrix(x);
			printf("expected_x: ");
			print_matrix(dataset.x);
			//print_matrix(dataset.A);
			//print_matrix(dataset.b);
		}
		Matrix* bx = aligned_vector(M, false);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < M; j++) {
				bx->elements[i] += A->elements[i * M + j] * x->elements[j];
			}
		}
		if (verbose) {
			printf("resulting b: ");
			print_matrix(bx);
		}
		all_rmse[k] = rmse(bx, b);
		printf("RMSE: %0.10f\n", all_rmse[k]);
		printf("iterations: %d\nseconds: %0.10f\n", iterations, seconds_spent);

		assert(x != last_x);

		free(bx->elements);
		free(x->elements);
		free(last_x->elements);
		free(dataset.x->elements);
		free(dataset.A->elements);
		free(dataset.b->elements);

		free(bx);
		free(x);
		free(last_x);
		free(dataset.x);
		free(dataset.A);
		free(dataset.b);
	}
	printf("Time: mean %0.10f std %0.10f\n", array_mean(exec_times, tests), array_std(exec_times, tests));
	printf("RMSE: mean %0.10f std %0.10f\n", array_mean(all_rmse, tests), array_std(all_rmse, tests));
	free(all_rmse);
	free(exec_times);

	return 0;
}
