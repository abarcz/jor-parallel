#ifndef _AMATRIX_H_
#define _AMATRIX_H_

#include <stdbool.h>
#include <stdlib.h>

/**
 *
 * Aligned matrix allocation and tools
 *
 * Author: Aleksy Barcz
 *
 */

#define ALIGN_BYTES 16

// choose floating point type for the library
typedef double fp_t;

/** generate random numer from [min, max] */
inline fp_t get_random(fp_t min, fp_t max)
{
	const fp_t range = max - min;
	return ((fp_t)rand() * range) / RAND_MAX + min;
}

/** allocate aligned memory block */
fp_t* aligned_vector(const int size, bool randomize);

/** allocate block of size = rows * rows */
fp_t* aligned_matrix(const int rows, bool randomize);

/** makes matrix diagonally dominant */
fp_t* make_diag_dominant(fp_t* matrix, const int rows);

/** multiply matrix * vector, newly allocated result is aligned */
fp_t* aligned_multiply(const fp_t* matrix, const fp_t* vector, const int rows);

/** A*x = b */
typedef struct DataSet
{
	fp_t* A;
	fp_t* b;
	fp_t* x;
} DataSet;

DataSet generate_dataset(const int rows);

/** return the absolute scalar differene between arrays */
fp_t array_diff(const fp_t a[], const fp_t b[], const int size);

fp_t array_mean(const fp_t* a, const int size);

fp_t array_std(const fp_t* a, const int size);

/** print array to stdout */
void print_array(const fp_t a[], const int size);

/** calculate RMSE between two arrays */
fp_t rmse(const fp_t a[], const fp_t b[], const int size);

#endif
