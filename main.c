
#include "amatrix.h"
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

void swap(double** a, double** b)
{
	double* c = *a;
	*a = *b;
	*b = c;
}

int main(int argc, char* argv[])
{
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
	const double min_diff = 0.000001;
	const double alpha = 1.01;

	double* exec_times = malloc(tests * sizeof(double));
	double* all_rmse = malloc(tests * sizeof(double));
	for (int k = 0; k < tests; k++) {

		const DataSet dataset = generate_dataset(M);

		double* last_x = aligned_vector(M, false);
		double* x = aligned_vector(M, false);
		for (int i = 0; i < M; i++) {
			x[i] = 1;
		}

		int iterations = 0;

		// solve Ax = b
		const double start_time = omp_get_wtime();

		double sum = 0;
		int j = 0;
		int i = 0;
		const double* A = dataset.A;
		const double* b = dataset.b;
		assert(x != last_x);
		#pragma omp parallel shared(last_x, x, iterations) private(i, j, sum)
		while (array_diff(x, last_x, M) > min_diff) {
			#pragma omp single
			{
				swap(&last_x, &x);
			}

			// A, M, alpha and b are constant, so they cannot be declared as shared
			#pragma omp for schedule(dynamic)
			for (i = 0; i < M; i++) {
				sum = 0;

				//#pragma omp target map(to: A, last_x) map(from: sum)a
				#pragma omp simd reduction(+: sum) aligned(A, last_x: 16) linear(j)
				for (j = 0; j < M; j++) {
					sum += A[i * M + j] * last_x[j];
				}

				sum -= A[i * M + i] * last_x[i];	// opt: outside the loop for sse optimizer
				x[i] = (1 - alpha) * last_x[i] + alpha * (b[i] - sum) / A[i * M + i];
			}

			#pragma omp single nowait
			{
				iterations++;
			}
		}
		const double end_time = omp_get_wtime();
		const double seconds_spent = end_time - start_time;
		exec_times[k] = seconds_spent;

		if (verbose) {
			printf("diff: %f\n", array_diff(x, last_x, M));
			printf("x: ");
			print_array(x, M);
			printf("expected_x: ");
			print_array(dataset.x, M);
		}
		double* bx = aligned_vector(M, false);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < M; j++) {
				bx[i] += A[i * M + j] * x[j];
			}
		}
		if (verbose) {
			printf("resulting b: ");
			print_array(bx, M);
		}
		all_rmse[k] = rmse(bx, b, M);
		printf("RMSE: %0.10f\n", all_rmse[k]);
		printf("iterations: %d\nseconds: %0.10f\n", iterations, seconds_spent);

		assert(x != last_x);
		free(dataset.A);
		free(dataset.x);
		free(dataset.b);
		free(x);
		free(last_x);
	}
	printf("Time: mean %0.10f std %0.10f\n", array_mean(exec_times, tests), array_std(exec_times, tests));
	printf("RMSE: mean %0.10f std %0.10f\n", array_mean(all_rmse, tests), array_std(all_rmse, tests));
	free(all_rmse);
	free(exec_times);

	return 0;
}
