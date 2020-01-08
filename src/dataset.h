#ifndef _DATASET_H
#define _DATASET_H

#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "mathUtils.h"

using namespace std;

void readSparse(string fileName, int offset, int n, int *indices, float *values, int *markers, unsigned int bufferlen);

void writeTopK(std::string filename, int numQueries, int k, unsigned int *topK);

void readTopK(std::string filename, int numQueries, int k, unsigned int *topK);

void similarityMetric(int *queries_indice, float *queries_val, int *queries_marker,
					  int *bases_indice, float *bases_val, int *bases_marker, unsigned int *queryOutputs,
					  unsigned int numQueries, unsigned int topk, unsigned int availableTopk, int *nList,
					  int nCnt);

#endif