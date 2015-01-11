#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdbool.h>
#include <stdlib.h>

/**
 *
 * Common settings for both CUDA (C++) and program (C) files
 *
 * Author: Aleksy Barcz
 *
 */

#define ALIGN_BYTES 16

// choose floating point type for the library
typedef float fp_t;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	fp_t* elements;
} Matrix;

inline void free_matrix(Matrix* m)
{
	free(m->elements);
	free(m);
}

#endif