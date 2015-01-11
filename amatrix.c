#include "amatrix.h"

#include <malloc.h>
#include <math.h>

fp_t* aligned_vector(const int size, bool randomize)
{
	fp_t* v = memalign(ALIGN_BYTES, size * sizeof(fp_t));
	if (v == NULL) {
		printf("Error allocating memory!");
		exit(-1);
	}
	if (randomize) {
		for (int i = 0; i < size; i++) {
			v[i] = get_random(-5, 5);
		}
	} else {
		for (int i = 0; i < size; i++) {
			v[i] = 0;
		}
	}
	return v;
}

fp_t* aligned_matrix(const int rows, bool randomize)
{
	return aligned_vector(rows * rows, randomize);
}

fp_t get_abs_row_sum(fp_t* matrix, const int row, const int rows)
{
	fp_t row_sum = 0;
	for (int j = 0; j < rows; j++) {
		row_sum += fabs(matrix[row * rows + j]);
	}
	return row_sum;
}

fp_t get_max_row_sum(fp_t* matrix, const int rows)
{
	fp_t max_row_sum = 0;
	for (int i = 0; i < rows; i++) {
		const fp_t row_sum = get_abs_row_sum(matrix, i, rows);
		if (row_sum > max_row_sum) {
			max_row_sum = row_sum;
		}
	}
	return max_row_sum;
}

fp_t* make_diag_dominant(fp_t* matrix, const int rows)
{
	for (int i = 0; i < rows; i++) {
		const fp_t row_sum = get_abs_row_sum(matrix, i, rows);
		matrix[i * rows + i] += get_random(1, 1.5) * (row_sum - fabs(matrix[i * rows + i]));
		if (get_random(-1, 1) < 0) {
			matrix[i * rows + i] = -matrix[i * rows + i];
		}
	}
	return matrix;
}

fp_t* aligned_multiply(const fp_t* matrix, const fp_t* vector, const int rows)
{
	fp_t* b = aligned_vector(rows, false);
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

fp_t array_diff(const fp_t a[], const fp_t b[], const int size)
{
	fp_t diff = 0;
	for (int i = 0; i < size; i++) {
		diff += fabs(a[i] - b[i]);
	}
	return diff;
}

fp_t array_mean(const fp_t* a, const int size)
{
	fp_t mean = 0;
	for (int i = 0; i < size; i++) {
		mean += a[i];
	}
	mean /= size;
	return mean;
}

fp_t array_std(const fp_t* a, const int size)
{
	const fp_t mean = array_mean(a, size);
	fp_t acc = 0;
	for (int i = 0; i < size; i++) {
		acc += pow(a[i] - mean, 2);
	}
	return sqrt(acc);
}

void print_array(const fp_t a[], const int size)
{
	for (int i = 0; i < size; i++) {
		printf("%f, ", a[i]);
	}
	printf("\n");
}

fp_t rmse(const fp_t a[], const fp_t b[], const int size)
{
	fp_t res = 0;
	for (int i = 0; i < size; i++) {
		res += pow((a[i] - b[i]), 2);
	}
	res = sqrt(res);
	return res;
}

