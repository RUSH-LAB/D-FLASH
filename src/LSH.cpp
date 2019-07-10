#include "LSH.h"

void LSH::getHashes(unsigned int *hashIndices, unsigned int *probeDataIdx, int *dataIdx, int *dataMarker, int numInputEntries, int numProbes) {

#pragma omp parallel for
	for (int inputIdx = 0; inputIdx < numInputEntries; inputIdx++) {

		unsigned int *hashes = new unsigned int[_numhashes];
		int sizenonzeros = dataMarker[inputIdx + 1] - dataMarker[inputIdx];
		
		optimalMinHash(hashes, (unsigned int*)(dataIdx + dataMarker[inputIdx]), sizenonzeros);

		for (int tb = 0; tb < _L; tb++)
		{
			unsigned int index = 0;
			for (int k = 0; k < _K; k++)
			{
				unsigned int h = hashes[_K*tb + k];
				h *= _rand1[_K * tb + k];
				h ^= h >> 13;
				h ^= _rand1[_K * tb + k];
				index += h*hashes[_K * tb + k];
			}
			index = (index << 2) >> (32 - _rangePow);

			hashIndices[hashIndicesOutputIdx(_L, numProbes, numInputEntries, inputIdx, 0, tb)] = index;
			probeDataIdx[hashIndicesOutputIdx(_L, numProbes, numInputEntries, inputIdx, 0, tb)] = inputIdx;
			for (int k = 1; k < numProbes; k++) {
				hashIndices[hashIndicesOutputIdx(_L, numProbes, numInputEntries, inputIdx, k, tb)] = index ^ (1 << (k - 1));
				probeDataIdx[hashIndicesOutputIdx(_L, numProbes, numInputEntries, inputIdx, k, tb)] = inputIdx;
			}
		}
		delete[] hashes;
	}
}