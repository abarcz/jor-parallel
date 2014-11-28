#ifndef _AMATRIX_H_
#define _AMATRIX_H_

#include <stdbool.h>

/**
 *
 * Aligned matrix allocation and tools
 *
 * Author: Aleksy Barcz
 *
 */

#define ALIGN_BYTES 16

/** allocate aligned memory block */
double* aligned_vector(const int size, bool randomize);

/** allocate block of size = rows * rows */
double* aligned_matrix(const int rows, bool randomize);

/** makes matrix diagonally dominant */
double* make_diag_dominant(double* matrix, const int rows);

/** multiply matrix * vector, newly allocated result is aligned */
double* aligned_multiply(const double* matrix, const double* vector, const int rows);

/** A*x = b */
typedef struct DataSet
{
	double* A;
	double* b;
	double* x;
} DataSet;

DataSet generate_dataset(const int rows);

/** return the absolute scalar differene between arrays */
double array_diff(const double a[], const double b[], const int size);

double array_mean(const double* a, const int size);

double array_std(const double* a, const int size);

/** print array to stdout */
void print_array(const double a[], const int size);

/** calculate RMSE between two arrays */
double rmse(const double a[], const double b[], const int size);

#endif
