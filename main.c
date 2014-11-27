#include <omp.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>

#define ALIGN_BYTES 16

/**
 *
 * Jacobi Over Relaxation using OpenMP
 *
 * Author: Aleksy Barcz
 *
 */

double* aligned_vector(const int size, bool randomize)
{
	double* v = memalign(ALIGN_BYTES, size * sizeof(double));
	if (v == NULL) {
		printf("Error allocating memory!");
		exit(-1);
	}
	if (randomize) {
		for (int i = 0; i < size; i++) {
			v[i] = (double)rand() / RAND_MAX;
		}
	} else {
		for (int i = 0; i < size; i++) {
			v[i] = 0;
		}
	}
	return v;
}

double* aligned_matrix(const int size, bool randomize)
{
	return aligned_vector(size * size, randomize);
}

double get_max_row_sum(double* matrix, const int rows)
{
	double max_row_sum = 0;
	for (int i = 0; i < rows; i++) {
		double row_sum = 0;
		for (int j = 0; j < rows; j++) {
			row_sum += matrix[i * rows + j];
		}
		if (row_sum > max_row_sum) {
			max_row_sum = row_sum;
		}
	}
	return max_row_sum;
}

double* make_diag_dominant(double* matrix, const int rows)
{
	const double max_row_sum = get_max_row_sum(matrix, rows);
	for (int i = 0; i < rows; i++) {
		matrix[i * rows + i] += (1 + rand() / RAND_MAX) * max_row_sum;
	}
	return matrix;
}

double* aligned_multiply(double* matrix, double* vector, const int rows)
{
	double* b = aligned_vector(rows, false);
	const int size = rows;
	for (int i = 0; i < size; i++) {
		b[i] = 0;
		for (int j = 0; j < size; j++) {
			b[i] += matrix[i * size + j] * vector[j];
		}
	}
	return b;
}

/** A*x = b */
typedef struct DataSet
{
	double* A;
	double* b;
	double* x;
} DataSet;

DataSet generate_dataset(const int rows)
{
	DataSet dataset;
	dataset.A = make_diag_dominant(aligned_matrix(rows, true), rows);
	dataset.x = aligned_vector(rows, true);
	dataset.b = aligned_multiply(dataset.A, dataset.x, rows);
	return dataset;
}

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

int main (int argc, char* argv[])
{
	srand(time(NULL));

	const bool verbose = false;
	//omp_set_dynamic(true);
	//omp_set_nested(true);
	//omp_set_num_threads(6);

	const int M = 1024;
	const DataSet dataset = generate_dataset(M);

	double* last_x = __builtin_assume_aligned(aligned_vector(M, false), ALIGN_BYTES);
	double* x = __builtin_assume_aligned(aligned_vector(M, false), ALIGN_BYTES);
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
	const double* A = __builtin_assume_aligned(dataset.A, ALIGN_BYTES);
	const double* b = __builtin_assume_aligned(dataset.b, ALIGN_BYTES);
	//#pragma omp parallel shared(last_x, x) private(i, j, sum)
	while (array_diff(x, last_x, M) > min_diff) {
		swap(&last_x, &x);

		// A, M, alpha and b are constant, so they cannot be declared as shared
		//#pragma omp for schedule(dynamic)
		for (i = 0; i < M; i++) {
			sum = 0;

			//#pragma omp parallel for shared(last_x) private(j) reduction(+: sum) schedule(dynamic)
			//#pragma omp target map(to: A, last_x) map(from: sum)
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

