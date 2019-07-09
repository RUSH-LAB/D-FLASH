
#include "indexing.h"
#include "misc.h"
#include "LSHReservoirSampler.h"

//#define PRINT_CLINFO

LSHReservoirSampler::LSHReservoirSampler(LSH *hashFamIn, unsigned int numHashPerFamily, unsigned int numHashFamilies,
	unsigned int reservoirSize, unsigned int dimension, unsigned int numSecHash, unsigned int maxSamples,
	unsigned int queryProbes, unsigned int hashingProbes, float tableAllocFraction, int myRank, int worldSize) {

#if !defined SECONDARY_HASHING
	if (numHashPerFamily != numSecHash) {
		std::cout << "[LSHReservoirSampler::LSHReservoirSampler] Fatal, secondary hashing disabled. " << std::endl;
	}
#endif

	_myRank = myRank;
	_worldSize = worldSize;

	initVariables(numHashPerFamily, numHashFamilies, reservoirSize, dimension, numSecHash, maxSamples, queryProbes,
		hashingProbes, tableAllocFraction);

	_hashFamily = hashFamIn;

	initHelper(_numTables, _rangePow, _reservoirSize);

	std::cout << "LSH Reservoir Initialized in Node " << _myRank << std::endl;
}


void LSHReservoirSampler::restart(LSH *hashFamIn, unsigned int numHashPerFamily, unsigned int numHashFamilies,
	unsigned int reservoirSize, unsigned int dimension, unsigned int numSecHash, unsigned int maxSamples,
	unsigned int queryProbes, unsigned int hashingProbes, float tableAllocFraction) {
	unInit();
	initVariables(numHashPerFamily, numHashFamilies, reservoirSize, dimension, numSecHash, maxSamples, queryProbes,
		hashingProbes, tableAllocFraction);
	_hashFamily = hashFamIn;
	initHelper(_numTables, _rangePow, _reservoirSize);
}

void LSHReservoirSampler::initVariables(unsigned int numHashPerFamily, unsigned int numHashFamilies,
	unsigned int reservoirSize, unsigned int dimension, unsigned int numSecHash, unsigned int maxSamples,
	unsigned int queryProbes, unsigned int hashingProbes, float tableAllocFraction) {
	_rangePow = numHashPerFamily;
	_numTables = numHashFamilies;
	_reservoirSize = reservoirSize;
	_dimension = dimension;
	_numSecHash = numSecHash;
	_maxSamples = maxSamples;
	_queryProbes = queryProbes;
	_hashingProbes = hashingProbes;
	_tableAllocFraction = tableAllocFraction;
	_segmentSizeModulor = numHashFamilies * reservoirSize * queryProbes - 1;
	_segmentSizeBitShiftDivisor = getLog2(_segmentSizeModulor);

	_numReservoirs = (unsigned int) pow(2, _rangePow);		// Number of rows in each hashTable.
	_numReservoirsHashed = (unsigned int) pow(2, _numSecHash);		// Number of rows in each hashTable.
	_aggNumReservoirs = (unsigned int) _numReservoirsHashed * _tableAllocFraction;
	_maxReservoirRand = (unsigned int) ceil(maxSamples / 10); // TBD.

	_zero = 0;
	_zerof = 0.0;
	_tableNull = TABLENULL;
}

void LSHReservoirSampler::initHelper(int numTablesIn, int numHashPerFamilyIn, int reservoriSizeIn) {

	srand(time(NULL));
	_sechash_a = rand() * 2 + 1;
	_sechash_b = rand();

	_global_rand = new unsigned int[_maxReservoirRand];

	_global_rand[0] = 0;
	for (int i = 1; i < _maxReservoirRand; i++) {
		_global_rand[i] = rand() % i;
	}


	/* Hash tables. */
	_tableMemReservoirMax = (_numTables - 1) * _aggNumReservoirs + _numReservoirsHashed;
	_tableMemMax = _tableMemReservoirMax * (1 + _reservoirSize);
	_tablePointerMax = _numTables * _numReservoirsHashed;

	_tableMem = new unsigned int[_tableMemMax]();
	_tableMemAllocator = new unsigned int[_numTables]();
	_tablePointers = new unsigned int[_tablePointerMax];
	_tablePointersLock = new omp_lock_t[_tablePointerMax];
	for (unsigned long long i = 0; i < _tablePointerMax; i++) {
		_tablePointers[i] = TABLENULL;
		omp_init_lock(_tablePointersLock + i);
	}
	_tableCountersLock = new omp_lock_t[_tableMemReservoirMax];
	for (unsigned long long i = 0; i < _tableMemReservoirMax; i++) {
		omp_init_lock(_tableCountersLock + i);
	}
	/* Hashing counter. */
	_sequentialIDCounter_kernel = 0;
}

LSHReservoirSampler::~LSHReservoirSampler() {

//	free(platforms); //For GPU??

	unInit();
}

void LSHReservoirSampler::unInit() {
	delete[] _tableMem;
	delete[] _tablePointers;
	delete[] _tableMemAllocator;
	for (unsigned long long i = 0; i < _tablePointerMax; i++) {
		omp_destroy_lock(_tablePointersLock + i);
	}
	for (unsigned long long i = 0; i < _tableMemReservoirMax; i++) {
		omp_destroy_lock(_tableCountersLock + i);
	}
	delete[] _tablePointersLock;
	delete[] _tableCountersLock;
	delete[] _global_rand;
}
