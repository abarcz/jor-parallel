#ifndef _AMATRIX_H_
#define _AMATRIX_H_

#include "common.h"
#include <stdbool.h>
#include <stdlib.h>

/**
 *
 * Aligned matrix allocation and tools
 *
 * Author: Aleksy Barcz
 *
 */

/** generate random numer from [min, max] */
inline fp_t get_random(fp_t min, fp_t max)
{
	const fp_t range = max - min;
	return ((fp_t)rand() * range) / RAND_MAX + min;
}

/** allocate aligned memory block */
Matrix* aligned_vector(const int size, bool randomize);

/** allocate block of size = rows * rows */
Matrix* aligned_matrix(const int rows, bool randomize);

/** makes matrix diagonally dominant */
Matrix* make_diag_dominant(Matrix* matrix);

/** multiply matrix * vector, newly allocated result is aligned */
Matrix* aligned_multiply(const Matrix* matrix, const Matrix* vector);

/** A*x = b */
typedef struct DataSet
{
	Matrix* A;
	Matrix* b;
	Matrix* x;
} DataSet;

inline void free_dataset(DataSet d)
{
	free_matrix(d.A);
	free_matrix(d.b);
	free_matrix(d.x);
}

DataSet generate_dataset(const int rows);

/** return the absolute scalar differene between arrays */
fp_t matrix_diff(const Matrix* a, const Matrix* b);

fp_t array_mean(const fp_t* a, const int size);

fp_t array_std(const fp_t* a, const int size);

/** print array to stdout */
void print_matrix(const Matrix* a);

/** calculate RMSE between two arrays */
fp_t rmse(const Matrix* a, const Matrix* b);

#endif
