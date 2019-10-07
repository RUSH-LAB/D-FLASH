#include "LSHReservoirSampler.h"
#include "misc.h"

void LSHReservoirSampler::add(int numInputEntries, int* dataIdx, float* dataVal, int* dataMarker, int dataOffset) {

	const int numProbePerTb = numInputEntries * _hashingProbes;

	if ((unsigned) numInputEntries > _maxSamples) {
		printf("[LSHReservoirSampler::add] Input length %d is too large! \n", numInputEntries);
		pause();
		return;
	}

	unsigned int* allprobsHash = new unsigned int[_numTables * numInputEntries * _hashingProbes];
	unsigned int* allprobsIdx = new unsigned int[_numTables * numInputEntries * _hashingProbes];

	_hashFamily->getHashes(allprobsHash, allprobsIdx, dataIdx, dataMarker, numInputEntries, _hashingProbes);

	unsigned int* storelog = new unsigned int[_numTables * 4 * numProbePerTb]();

	// for (int n = 0; n < _worldSize; n++) {
	// 	if (_myRank == n) {
	// 		printf("\nData Vector Hashes Node %d:\n", _myRank);
	// 		for (int i = 0; i < numInputEntries; i++) {
	// 			printf("\tVector %d: ", i);
	// 			for (int t = 0; t < _numTables; t++) {
	// 				printf("%d ", allprobsHash[t * numInputEntries + i]);
	// 			}
	// 			printf("\n\n");
	// 		}
	// 	}
	// 	MPI_Barrier(MPI_COMM_WORLD);
	// }

	reservoirSampling(allprobsHash, allprobsIdx, storelog, numProbePerTb);
	addTable(storelog, numProbePerTb, dataOffset);

	delete[] storelog;
	delete[] allprobsHash;
	delete[] allprobsIdx;

	_sequentialIDCounter_kernel += numInputEntries;
}

void LSHReservoirSampler::ann(int numQueryEntries, int* dataIdx, float* dataVal, int* dataMarker, unsigned int* outputs, int topk) {

	if ((unsigned) topk > _reservoirSize * _numTables) {
		printf("Error: Maximum k exceeded! %d\n", topk);
		pause();
		return;
	}

	unsigned int* allprobsHash = new unsigned int[_numTables * numQueryEntries * _queryProbes];
	unsigned int* allprobsIdx = new unsigned int[_numTables * numQueryEntries * _queryProbes];
	int segmentSize = _numTables * _queryProbes * _reservoirSize;

	_hashFamily->getHashes(allprobsHash, allprobsIdx,
		dataIdx, dataMarker, numQueryEntries, _queryProbes);

	unsigned int* tally = new unsigned int[numQueryEntries * segmentSize];

	extractReservoirs(numQueryEntries, segmentSize, tally, allprobsHash);

	kSelect(tally, outputs, segmentSize, numQueryEntries, topk);

	delete[] allprobsHash;
	delete[] allprobsIdx;
	delete[] tally;
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


// For Comparison with K Select with TopKAPI, Testing Only
void LSHReservoirSampler::extractTopK(int numQueryEntries, unsigned int* hashIndices, int topK, unsigned int* outputs) {
	int segmentSize = _numTables * _queryProbes * _reservoirSize;
	unsigned int* tally = new unsigned int[numQueryEntries * segmentSize];
	extractReservoirs(numQueryEntries, segmentSize, tally, hashIndices);
	kSelect(tally, outputs, segmentSize, numQueryEntries, topK);
}

void LSHReservoirSampler::kSelect(unsigned int *tally, unsigned int *outputs, int segmentSize, int numQueryEntries, int topk) {
	// SegmentedSort.
#pragma omp parallel for
	for (int i = 0; i < numQueryEntries; i++) {
		std::sort(tally + i * segmentSize, tally + i * segmentSize + segmentSize);
	}

	// Reduction.
	unsigned int *tallyCnt = new unsigned int[segmentSize * numQueryEntries]();

#if !defined SINGLETHREAD_COUNTING
#pragma omp parallel for
#endif

	for (int i = 0; i < numQueryEntries; i++) {
		unsigned int *vec = tally + i * segmentSize;
		unsigned int *cntvec = tallyCnt + i * segmentSize;
		int prev = vec[0];
		int ct = 0;
		int counter = 0;
		for (int j = 1; j < segmentSize; j++) {
			counter++;
			if (prev != vec[j]) {
				vec[ct] = prev;
				cntvec[ct] = counter;
				prev = vec[j];
				counter = 0;
				ct++;
			}
		}
		for (; ct < segmentSize; ct++) {
			vec[ct] = 0;
		}
	}
	// KV SegmentedSort.
#pragma omp parallel for
	for (int i = 0; i < numQueryEntries; i++) {
		unsigned int *vec = tally + i * segmentSize;
		unsigned int *cntvec = tallyCnt + i * segmentSize;
		unsigned int *idx = new unsigned int[segmentSize];
		for (int j = 0; j < segmentSize; j++) {
			idx[j] = j;
		}
		std::sort(idx, idx + segmentSize,
			[&cntvec](unsigned int i1, unsigned int i2) { return cntvec[i1] > cntvec[i2]; });

		int ss;
		int ct = 0;
		if (vec[idx[0]] == 0) { // The first item is spurious.
			ss = 1;
		}
		else {
			ss = 0;
		}
		ct = 0;
		for (int k = ss; k < topk + ss; k++) {
			outputs[i * topk + ct] = vec[idx[k]];
			ct++;
		}
		delete[] idx;
	}
	delete[] tallyCnt;
}