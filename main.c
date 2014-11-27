
#include "amatrix.h"
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

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

void swap(double** a, double** b)
{
	double* c = *a;
	*a = *b;
	*b = c;
}

int main(int argc, char* argv[])
{
	if (argc != 2) {
		printf("program must be called with matrix dimension as the only argument\n");
		exit(1);
	}
	const int M = atoi(argv[1]);
	printf("Using matrix dimension: %d\n", M);

	// always use the same seed to get the same matrices during tests
	srand(0);

	const bool verbose = false;

	const DataSet dataset = generate_dataset(M);

	double* last_x = aligned_vector(M, false);
	double* x = aligned_vector(M, false);
	for (int i = 0; i < M; i++) {
		x[i] = 1;
	}

	const double min_diff = 0.000001;
	const double alpha = 1.01;
	int iterations = 0;

	// solve Ax = b
	const double start_time = omp_get_wtime();

	double sum = 0;
	int j = 0;
	int i = 0;
	const double* A = dataset.A;
	const double* b = dataset.b;
	#pragma omp parallel shared(last_x, x) private(i, j, sum)
	while (array_diff(x, last_x, M) > min_diff) {
		swap(&last_x, &x);

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
		iterations++;
	}
	const double end_time = omp_get_wtime();
	const double seconds_spent = end_time - start_time;

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
	printf("RMSE: %0.10f\n", rmse(bx, b, M));
	printf("iterations: %d\nseconds: %0.10f\n", iterations, seconds_spent);

	return 0;
}
