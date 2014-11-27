#include "amatrix.h"

#include <stdlib.h>
#include <malloc.h>

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

double* aligned_matrix(const int rows, bool randomize)
{
	return aligned_vector(rows * rows, randomize);
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

double* aligned_multiply(const double* matrix, const double* vector, const int rows)
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

DataSet generate_dataset(const int rows)
{
	DataSet dataset;
	dataset.A = make_diag_dominant(aligned_matrix(rows, true), rows);
	dataset.x = aligned_vector(rows, true);
	dataset.b = aligned_multiply(dataset.A, dataset.x, rows);
	return dataset;
}
