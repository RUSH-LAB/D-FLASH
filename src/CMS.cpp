#include "CMS.h"


CMS::CMS(int L, int B, int numDataStreams, int myRank, int worldSize) {
	_numHashes = L;
	_bucketSize = B;
	_myRank = myRank;
	_worldSize = worldSize;
	_numSketches = numDataStreams;
	_sketchSize = _numHashes * _bucketSize * 2;
	_LHH = new int[_numSketches * _sketchSize]();
	_hashingSeeds = new unsigned int[_numHashes];

	if (_myRank == 0) {
		//Random hash functions
		//srand(time(NULL));

		//Fixed random seeds for hash functions
		srand(8524023);
		
		for (int h = 0; h < _numHashes; h++) {
			_hashingSeeds[h] = rand();
			if (_hashingSeeds[h] % 2 == 0) {
				_hashingSeeds[h]++;
			}
		}
	}

	MPI_Bcast(_hashingSeeds, _numHashes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	std::cout << "CMS Initialized in Node " << _myRank << std::endl;
}

void CMS::getCanidateHashes(int candidate, unsigned int* hashes){
	for (int hashIndx = 1; hashIndx < _numHashes; hashIndx++) {
		unsigned int h = _hashingSeeds[hashIndx];
		unsigned int k = candidate;
		k *= 0xcc9e2d51;
		k = (k << 15) | (k >> 17);
		k *= 0x1b873593;
		h ^= k;
		h = (h << 13) | (h >> 19);
		h = h * 5 + 0xe6546b64;
		h ^= h >> 16;
		h *= 0x85ebca6b;
		h ^= h >> 13;
		h *= 0xc2b2ae35;
		h ^= h >> 16;
		unsigned int curhash = (unsigned int) h % _bucketSize;
		hashes[hashIndx] = curhash;
	}
}

void CMS::getHashes(unsigned int* dataStream, int dataStreamLen, unsigned int* hashIndices) {

#pragma omp parallel for default (none) shared(dataStream, hashIndices, dataStreamLen)
	for (int dataIndx = 0; dataIndx < dataStreamLen; dataIndx++) {
		for (int hashIndx = 0; hashIndx < _numHashes; hashIndx++) {
			unsigned int h = _hashingSeeds[hashIndx];
			unsigned int k = (unsigned int) dataStream[dataIndx];
			k *= 0xcc9e2d51;
			k = (k << 15) | (k >> 17);
			k *= 0x1b873593;
			h ^= k;
			h = (h << 13) | (h >> 19);
			h = h * 5 + 0xe6546b64;
			h ^= h >> 16;
			h *= 0x85ebca6b;
			h ^= h >> 13;
			h *= 0xc2b2ae35;
			h ^= h >> 16;
			unsigned int curhash = (unsigned int) h % _bucketSize;
			hashIndices[hashLocation(dataIndx, _numHashes, hashIndx)] = curhash;
		}
	}
}

void CMS::addSketch(int dataStreamIndx, unsigned int* dataStream, int dataStreamLen) {

	unsigned int* hashIndices = new unsigned int[_numHashes * dataStreamLen];
	getHashes(dataStream, dataStreamLen, hashIndices);

	for (int dataIndx = 0; dataIndx < dataStreamLen; dataIndx++) {
		for (int hashIndx = 0; hashIndx < _numHashes; hashIndx++) {
			if (dataStream[dataIndx] == 0) {
				continue;
			}
			unsigned int currentHash = hashIndices[hashLocation(dataIndx, _numHashes, hashIndx)];
			int* LHH_ptr = _LHH + heavyHitterIndx(dataStreamIndx, _sketchSize, _bucketSize, hashIndx, currentHash);
			int* LHH_Count_ptr = _LHH + countIndx(dataStreamIndx, _sketchSize, _bucketSize, hashIndx, currentHash);
			if (*LHH_Count_ptr != 0) {
				if (dataStream[dataIndx] == *LHH_ptr) {
					*LHH_Count_ptr = *LHH_Count_ptr + 1;
				} else {
					*LHH_Count_ptr = *LHH_Count_ptr - 1;
				}
			}
			if (*LHH_Count_ptr == 0) {
				*LHH_ptr = (int) dataStream[dataIndx];
				*LHH_Count_ptr = 1;
			}
		}
	}
}

void CMS::add(unsigned int* dataStreams, int segmentSize) {

#pragma omp parallel for default(none) shared(dataStreams, segmentSize)
	for (int streamIndx = 0; streamIndx < _numSketches; streamIndx++) {
		addSketch(streamIndx, dataStreams + streamIndx * segmentSize, segmentSize);
	}
}


void CMS::topKSketch(int K, int threshold, unsigned int* topK, int sketchIndx) {

	LHH* candidates = new LHH[_bucketSize];
	int count = 0;
	for (int b = 0; b < _bucketSize; b++) {
		int currentHeavyHitter = _LHH[heavyHitterIndx(sketchIndx, _sketchSize, _bucketSize, 0, b)];
		int currentCount = _LHH[countIndx(sketchIndx, _sketchSize, _bucketSize, 0, b)];
		if (currentCount >= threshold) {
			candidates[count].heavyHitter = currentHeavyHitter;
			candidates[count].count = currentCount;
			count++;
		} else {
			unsigned int* hashes = new unsigned int[_numHashes];
			getCanidateHashes(currentHeavyHitter, hashes);
			for (int hashIndx = 1; hashIndx < _numHashes; hashIndx++) {
				currentCount = _LHH[countIndx(sketchIndx, _sketchSize, _bucketSize, hashIndx, hashes[hashIndx])];
				if (currentCount > threshold) {
					candidates[count].heavyHitter = currentHeavyHitter;
					candidates[count].count = currentCount;
					count++;
					break;
				}
			}
		}
	}
	for (; count < _bucketSize; count++) {
		candidates[count].heavyHitter = -1;
		candidates[count].count = -1;
	}
	std::sort(candidates, candidates + _bucketSize, [&candidates](LHH a, LHH b){return a.count > b.count;});

	for (int i = 0; i < K; i++) {
		if (candidates[i].heavyHitter > -1) {
			topK[i] = candidates[i].heavyHitter;
		}
	}
	delete[] candidates;
}

void CMS::topK (int topK, unsigned int* outputs, int threshold) {

#pragma omp parallel for default(none) shared(topK, outputs, threshold)
    for (int sketchIndx = 0; sketchIndx < _numSketches; sketchIndx++) {
         topKSketch(topK, threshold, outputs + sketchIndx * topK, sketchIndx);
    }
}

void CMS::combineSketches (int* newLHH) {

// #pragma omp parallel for default(none) shared(newLHH, _LHH, _numHashes, _bucketSize, _numSketches)
#pragma omp parallel for default(none) shared(newLHH)
	for (int n = 0; n < _numHashes * _bucketSize * _numSketches; n++) {
		if (newLHH[n * 2] == _LHH[n * 2]) {
			_LHH[n * 2 + 1] += newLHH[n * 2 + 1];
		} else {
			_LHH[n * 2 + 1] -= newLHH[n * 2 + 1];
		}
		if (_LHH[n * 2 + 1] <= 0) {
			_LHH[n * 2] = newLHH[n * 2];
			_LHH[n * 2 + 1] = - _LHH[n * 2 + 1];
		}
	}
}

void CMS::aggregateSketches() {
	int bufferSize = _sketchSize * _numSketches;
	int* sketchBuffer;
	if (_myRank == 0) {
		sketchBuffer = new int[bufferSize * _worldSize];
	}
	MPI_Gather(_LHH, bufferSize, MPI_INT, sketchBuffer, bufferSize, MPI_INT, 0, MPI_COMM_WORLD);
	if (_myRank == 0) {
		for (int n = 1; n < _worldSize; n++) {
			combineSketches(sketchBuffer + (n * bufferSize));
		}
		delete[] sketchBuffer;
	}
}


void CMS::showCMS(int sketchIndx) {
	for (int l = 0; l < _numHashes; l++) {
		printf("Bucket %d:\n\t[LHH]: ", l);
		for (int b = 0; b < _bucketSize; b++) {
			printf("\t%d", _LHH[heavyHitterIndx(sketchIndx, _sketchSize, _bucketSize, l, b)]);
		}
		printf("\n\t[Cnt]: ");
		for (int b = 0; b < _bucketSize; b++) {
			printf("\t%d", _LHH[countIndx(sketchIndx, _sketchSize, _bucketSize, l, b)]);
		}
		printf("\n");
	}
}

CMS::~CMS() {

	delete[] _LHH;
	delete[] _hashingSeeds;
}
