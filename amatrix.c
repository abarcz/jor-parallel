#include "amatrix.h"

#include <malloc.h>
#include <math.h>
#include <assert.h>

Matrix* aligned_vector(const int size, bool randomize)
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
	Matrix* m = malloc(sizeof(Matrix));
	m->width = size;
	m->height = 1;
	m->elements = v;
	return m;
}

Matrix* aligned_matrix(const int rows, bool randomize)
{
	Matrix* m = aligned_vector(rows * rows, randomize);
	m->height = rows;
	m->width = rows;
	return m;
}

fp_t get_abs_row_sum(const Matrix* matrix, const int row)
{
	fp_t row_sum = 0;
	for (int j = 0; j < matrix->width; j++) {
		row_sum += fabs(matrix->elements[row * matrix->width + j]);
	}
	return row_sum;
}

fp_t get_max_row_sum(const Matrix* matrix)
{
	fp_t max_row_sum = 0;
	for (int i = 0; i < matrix->width; i++) {
		const fp_t row_sum = get_abs_row_sum(matrix, i);
		if (row_sum > max_row_sum) {
			max_row_sum = row_sum;
		}
	}
	return max_row_sum;
}

Matrix* make_diag_dominant(Matrix* matrix)
{
	for (int i = 0; i < matrix->width; i++) {
		const fp_t row_sum = get_abs_row_sum(matrix, i);
		const int index = i * matrix->width + i;
		matrix->elements[index] += get_random(1, 1.5) * (row_sum - fabs(matrix->elements[index]));
		if (get_random(-1, 1) < 0) {
			matrix->elements[index] = -matrix->elements[index];
		}
	}
	return matrix;
}

Matrix* aligned_multiply(const Matrix* matrix, const Matrix* vector)
{
	assert(matrix->width == vector->width);
	Matrix* b = aligned_vector(matrix->width, false);
	const int size = matrix->width;
	for (int i = 0; i < size; i++) {
		b->elements[i] = 0;
		for (int j = 0; j < size; j++) {
			b->elements[i] += matrix->elements[i * size + j] * vector->elements[j];
		}
	}
	return b;
}

DataSet generate_dataset(const int rows)
{
	DataSet dataset;
	dataset.A = make_diag_dominant(aligned_matrix(rows, true));
	dataset.x = aligned_vector(rows, true);
	dataset.b = aligned_multiply(dataset.A, dataset.x);
	return dataset;
}

fp_t matrix_diff(const Matrix* a, const Matrix* b)
{
	assert(a->width == b->width);
	assert(a->height == b->height);
	const int size = a->width;
	fp_t diff = 0;
	for (int i = 0; i < size; i++) {
		diff += fabs(a->elements[i] - b->elements[i]);
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

void print_matrix(const Matrix* a)
{
	for (int i = 0; i < a->width; i++) {
		printf("%f, ", a->elements[i]);
	}
	printf("\n");
}

fp_t rmse(const Matrix* a, const Matrix* b)
{
	assert(a->width == b->width);
	assert(a->height == b->height);
	fp_t res = 0;
	for (int i = 0; i < a->width; i++) {
		res += pow((a->elements[i] - b->elements[i]), 2);
	}
	res = sqrt(res);
	return res;
}

