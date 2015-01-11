
#include "amatrix.h"
#include "deviceid.h"
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

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

extern int cuda_identify();
extern void cuda_add(float* A, float* B, float* C);

/** call: ./main <matrix_dimension> <number_of_tests> */
int main(int argc, char* argv[])
{
	cuda_identify();
	size_t size = N * sizeof(float);

	// Allocate input vectors h_A and h_B in host memory
	float* h_A = (float*)malloc(size);
	float* h_B = (float*)malloc(size);
	float* h_C = (float*)malloc(size);
	h_A[0] = 1;
	h_A[1] = 2;
	h_B[0] = 3;
	h_B[1] = 4;
	h_C[0] = 0;
	h_C[1] = 0;
	cuda_add(h_A, h_B, h_C);
	for (int i = 0; i < N ; i++)
		printf("%f ", h_C[i]);
	printf("\n");


	if (argc != 3) {
		printf("program must be called with arguments: matrix_dimension tests_number\n");
		exit(1);
	}
	const int M = atoi(argv[1]);
	printf("Using matrix dimension: %d\n", M);
	const int tests = atoi(argv[2]);

	// always use the same seed to get the same matrices during tests
	srand(0);

	const bool verbose = false;
	const fp_t min_diff = 0.00001;
	const fp_t alpha = 0.9;
	const int max_iter = 200;

	fp_t* exec_times = malloc(tests * sizeof(fp_t));
	fp_t* all_rmse = malloc(tests * sizeof(fp_t));
	for (int k = 0; k < tests; k++) {

		const DataSet dataset = generate_dataset(M);

		Matrix* last_x = aligned_vector(M, false);
		Matrix* x = aligned_vector(M, false);
		for (int i = 0; i < M; i++) {
			x->elements[i] = 1;
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
		//fp_t* sumv = aligned_vector(M, false);
		//#pragma omp parallel shared(last_x, x, iterations) private(i, j, sum)
		while ((matrix_diff(x, last_x) > min_diff) && (max_iter < 0 || iterations < max_iter)) {
			//#pragma omp single
			{
				swap(last_x, x);
			}

			// A, M, alpha and b are constant, so they cannot be declared as shared
			//#pragma omp for schedule(dynamic)
			for (i = 0; i < M; i++) {
				sum = 0;

				//#pragma omp target map(to: A, last_x) map(from: sum)
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
		}
		const fp_t end_time = omp_get_wtime();
		const fp_t seconds_spent = end_time - start_time;
		exec_times[k] = seconds_spent;

		if (verbose) {
			printf("diff: %f\n", matrix_diff(x, last_x));
			printf("x: ");
			print_matrix(x);
			printf("expected_x: ");
			print_matrix(dataset.x);
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
		free_dataset(dataset);
		free_matrix(x);
		free_matrix(last_x);
	}
	printf("Time: mean %0.10f std %0.10f\n", array_mean(exec_times, tests), array_std(exec_times, tests));
	printf("RMSE: mean %0.10f std %0.10f\n", array_mean(all_rmse, tests), array_std(all_rmse, tests));
	free(all_rmse);
	free(exec_times);

	return 0;
}
