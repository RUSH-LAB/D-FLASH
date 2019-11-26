#include "LSHReservoirSampler.h"

void LSHReservoirSampler::add(int numInputEntries, int* dataIdx, float* dataVal, int* dataMarker, int dataOffset) {

	const int numProbePerTb = numInputEntries * _hashingProbes;

	if ((unsigned) numInputEntries > _maxSamples) {
		printf("[LSHReservoirSampler::add] Input length %d is too large! \n", numInputEntries);
		return;
	}

	unsigned int* allprobsHash = new unsigned int[_numTables * numInputEntries * _hashingProbes];
	unsigned int* allprobsIdx = new unsigned int[_numTables * numInputEntries * _hashingProbes];

	_hashFamily->getHashes(allprobsHash, allprobsIdx, dataIdx, dataMarker, numInputEntries, _hashingProbes);

	unsigned int* storelog = new unsigned int[_numTables * 4 * numProbePerTb]();

	reservoirSampling(allprobsHash, allprobsIdx, storelog, numProbePerTb);
	addTable(storelog, numProbePerTb, dataOffset);

	delete[] storelog;
	delete[] allprobsHash;
	delete[] allprobsIdx;

	_sequentialIDCounter_kernel += numInputEntries;
}

void LSHReservoirSampler::extractReservoirs(int numQueryEntries, int segmentSize, unsigned int *queue, unsigned int *hashIndices) {

	unsigned int hashIdx, allocIdx;
#pragma omp parallel for private(hashIdx, allocIdx)
	for (int tb = 0; tb < _numTables; tb++) {
		for (int queryIdx = 0; queryIdx < numQueryEntries; queryIdx++) {
			for (int elemIdx = 0; elemIdx < _reservoirSize; elemIdx++) {
				for (unsigned int k = 0; k < _queryProbes; k++) {
					hashIdx = hashIndices[allprobsHashIdx(_queryProbes, numQueryEntries, tb, queryIdx, k)];
					allocIdx = _tablePointers[tablePointersIdx(_numReservoirsHashed, hashIdx, tb, _sechash_a, _sechash_b)];
					if (allocIdx != TABLENULL) {
						queue[queueElemIdx(segmentSize, tb, queryIdx, k, elemIdx)] =
							_tableMem[tableMemResIdx(tb, allocIdx, _aggNumReservoirs) + elemIdx];
					}
				}
			}
		}
	}
}

void LSHReservoirSampler::getQueryHash(int queryPartitionSize, int numQueryPartitionHashes, int* queryPartitionIndices, float* queryPartitionVals,
								 	   int* queryPartitionMarkers, unsigned int* queryHashes) {

	unsigned int* allprobsIdx = new unsigned int[numQueryPartitionHashes];

	_hashFamily->getHashes(queryHashes, allprobsIdx, queryPartitionIndices, 
								   queryPartitionMarkers, queryPartitionSize, _queryProbes);

	delete[] allprobsIdx;
}
