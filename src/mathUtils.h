#ifndef _MATH_UTILS
#define _MATH_UTILS

#include <math.h>

int smallestPow2(int x);
void zCentering(float *values, int n);
unsigned int getLog2(unsigned int x);
float cosineDist(float *A, float *B, unsigned int n);
float cosineDist(int *indiceA, float *valA, int nonzerosA, int *indiceB, float *valB, int nonzerosB);

float SparseVecMul(int *indicesA, float *valuesA, unsigned int sizeA,
	int *indicesB, float *valuesB, unsigned int sizeB);
float SparseVecMul(int *indicesA, float *valuesA, unsigned int sizeA, float *B);

#endif