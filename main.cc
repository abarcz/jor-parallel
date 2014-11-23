#include <omp.h>
#include <cstring>
#include <cmath>
#include <iostream>

#include "data32.h"


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
		diff += std::abs(a[i] - b[i]);
	}
	return diff;
}

/** print array to stdout */
void print_array(const double a[], const int size)
{
	for (int i = 0; i < size; i++) {
		std::cout << a[i] << ", ";
	}
	std::cout << std::endl;
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


int main (int argc, char * argv [])
{
	std::cout << "Go!" << std::endl;

	//const int M = 3;	// assume symmetric matrices
	//const double A[M][M] = {{4, -1, -1}, {-2, 6, 1}, {-1, 1, 7}};
	//const double b[M] = {3, 9, -6};
	double last_x[M] = {0};
	double x[M] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};


	const double min_diff = 0.0001;
	const double alpha = 1;
	int iterations = 0;

	// solve Ax = b
	const double start_time = omp_get_wtime();
	while (array_diff(x, last_x, M) > min_diff) {
		memcpy(last_x, x, sizeof(x));
		for (int i = 0; i < M; i++) {
			double sum = 0;
			for (int j = 0; j < M; j++) {
				if (j != i) {
					sum += A[i][j] * last_x[j];
				}
			}
			x[i] = (1 - alpha) * last_x[i] + alpha * (b[i] - sum) / A[i][i];
		}
		iterations++;
	}
	const double end_time = omp_get_wtime();
	const double seconds_spent = end_time - start_time;


	std::cout << "diff: " << array_diff(x, last_x, M) << std::endl;
	std::cout << "x: ";
	print_array(x, M);
	double bx[M] = {0};
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			bx[i] += A[i][j] * x[j];
		}
	}
	std::cout << "resulting b: ";
	print_array(bx, M);
	std::cout << "RMSE: " << rmse(bx, b, M) << std::endl;
	std::cout << "result obtained after " << iterations << " iterations, " << seconds_spent << "s" << std::endl;

	return 0;
}

