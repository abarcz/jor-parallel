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
#define DOUBLE
#ifdef DOUBLE
	typedef double fp_t;
#else
	typedef float fp_t;
	// choose texture usage
	#define TEXTURE
#endif

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	int size;
	fp_t* elements;
} Matrix;

#endif
