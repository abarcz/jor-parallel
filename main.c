#include <omp.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#include "data32.h"


/**
 *
 * Jacobi Over Relaxation using OpenMP
 *
 * Author: Aleksy Barcz
 *
 */

/** return the absolute scalar differene between arrays */
double array_diff(const double a[], const double b[], const int size)
{
	double diff = 0;
	for (int i = 0; i < size; i++){
		diff += fabs(a[i] - b[i]);
	}
	return diff;
}

/** print array to stdout */
void print_array(const double a[], const int size)
{
	for (int i = 0; i < size; i++) {
		printf("%f, ", a[i]);
	}
	printf("\n");
}

/** calculate RMSE between two arrays */
double rmse(const double a[], const double b[], const int size)
{
	double res = 0;
	for (int i = 0; i < size; i++) {
		res += pow((a[i] - b[i]), 2);
	}
	res = sqrt(res);
	return res;
}


int main (int argc, char * argv [])
{
	const bool verbose = false;
	//omp_set_dynamic(true);
	//omp_set_nested(true);
	//omp_set_num_threads(6);

	//const int M = 3;	// assume symmetric matrices
	//const double A[M][M] = {{4, -1, -1}, {-2, 6, 1}, {-1, 1, 7}};
	//const double b[M] = {3, 9, -6};
	double last_x[M] __attribute__((aligned(16))) = {0};
	double x[M] __attribute__((aligned(16))) = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

	const double min_diff = 0.0001;
	const double alpha = 1.01;
	int iterations = 0;

	// solve Ax = b
	const double start_time = omp_get_wtime();
	while (array_diff(x, last_x, M) > min_diff) {
		memcpy(last_x, x, sizeof(x));

		// A, M, alpha and b are constant, so they cannot be declared as shared
		//#pragma omp parallel for shared(last_x, x) schedule(dynamic)
		for (int i = 0; i < M; i++) {
			double sum = 0;
			int j = 0;

			// omp simd doesn't generate any addpd or mulpd instructions, it tampers the -O2 auto-vectorisation
			//#pragma omp simd reduction(+: sum) aligned(A, last_x: 16) linear(j)
			//#pragma omp parallel for shared(last_x) private(j) reduction(+: sum) schedule(dynamic)
			for (j = 0; j < M; j++) {
				sum += A[i][j] * last_x[j];
			}

			sum -= A[i][i] * last_x[i];	// opt: outside the loop for gcc sse optimizer
			x[i] = (1 - alpha) * last_x[i] + alpha * (b[i] - sum) / A[i][i];
		}
		iterations++;
	}
	const double end_time = omp_get_wtime();
	const double seconds_spent = end_time - start_time;

	if (verbose) {
		printf("diff: %f\n", array_diff(x, last_x, M));
		printf("x: ");
		print_array(x, M);
	}
	double bx[M] = {0};
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			bx[i] += A[i][j] * x[j];
		}
	}
	if (verbose) {
		printf("resulting b: ");
		print_array(bx, M);
	}
	printf("RMSE: %0.10f\n", rmse(bx, b, M));
	printf("iterations: %d, seconds: %0.10f\n", iterations, seconds_spent);

	return 0;
}

