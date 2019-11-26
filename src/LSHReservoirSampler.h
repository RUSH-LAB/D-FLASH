#ifndef _LSH_RESERVOIR_SAMPLER_H
#define _LSH_RESERVOIR_SAMPLER_H

#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <math.h>
#include "indexing.h"
#include "mathUtils.h"

#include "omp.h"
#include "LSH.h"

#define TABLE_NULL -1


#define DEBUGTB 3
#define DEBUGENTRIES 20

/** LSHReservoirSampler Class.
	Providing hashtable data-structure and k-select algorithm.
	An LSH class instantiation is pre-required.
*/
class LSHReservoirSampler {
private:

	int _myRank, _worldSize;

	LSH *_hashFamily;
	unsigned int _rangePow, _numTables, _reservoirSize, _dimension, _numSecHash, _maxSamples,
		_maxReservoirRand, _queryProbes, _hashingProbes, _segmentSizeModulor, _segmentSizeBitShiftDivisor;
	float _tableAllocFraction;

	unsigned int* _tableMem;
	unsigned int* _tableMemAllocator; // Special value MAX - 1.
	unsigned int* _tablePointers;
	omp_lock_t* _tablePointersLock;
	omp_lock_t* _tableCountersLock;

	unsigned int *_global_rand;
	unsigned int _numReservoirs, _sequentialIDCounter_kernel, _numReservoirsHashed, _aggNumReservoirs;
	unsigned long long _tableMemMax, _tableMemReservoirMax, _tablePointerMax;
	float _zerof;
	unsigned int _sechash_a, _sechash_b, _tableNull, _zero;

	/* Init. */
	void initVariables(unsigned int numHashPerFamily, unsigned int numHashFamilies, unsigned int reservoirSize,
		unsigned int dimension, unsigned int numSecHash, unsigned int maxSamples, unsigned int queryProbes,
		unsigned int hashingProbes, float tableAllocFraction);
	void initHelper(int numTablesIn, int numHashPerFamilyIn, int reservoriSizeIn);
	void unInit();

	// Samples reservoirs and determines where to add data vectors.
	void reservoirSampling(unsigned int *allprobsHash, unsigned int *allprobsIdx, unsigned int *storelog, int numProbePerTb);

	// Adds data vectors to tables at locations determined by reservoirSampling function. 
	// Param dataOffset is used to account for indexing across nodes.
	void addTable(unsigned int *storelog, int numProbePerTb, int dataOffset);

	/* Debug. */
	void viewTables();
	int benchCounting(int segmentSize, int* dataIdx, float* dataVal, int* dataMarker, float *timings);

public:
	void restart(LSH *hashFamIn, unsigned int numHashPerFamily, unsigned int numHashFamilies,
		unsigned int reservoirSize, unsigned int dimension, unsigned int numSecHash, unsigned int maxSamples,
		unsigned int queryProbes, unsigned int hashingProbes, float tableAllocFraction);

	/* Constructor.

	@param hashFam: an LSH class, a family of hash functions.
	@param numHashPerFamily: number of hashes (bits) per hash table, have to be the same as that of the hashFam.
	@param numHashFamilies: number of hash families (tables), have to be the same as that of the hashFam.
	@param reservoirSize: size of each hash rows (reservoir).
	@param dimension: for dense vectors, this is the dimensionality of each vector. 
		For sparse format data, this number is not used. (TBD)
	@param numSecHash: the number of secondary hash bits. A secondary (universal) hashing is used to shrink the
		original range of the LSH for better table occupancy. Only a number <= numHashPerFamily makes sense.
	@param maxSamples: the maximum number incoming data points to be hashed and added.
	@param queryProbes: number of probes per query per table.
	@param hashingProbes: number of probes per data point per table.
	@param tableAllocFraction: fraction of reservoirs to allocate for each table, will share with other table if overflows.
	*/
	LSHReservoirSampler(LSH *hashFam, unsigned int numHashPerFamily, unsigned int numHashFamilies,
		unsigned int reservoirSize, unsigned int dimension, unsigned int numSecHash, unsigned int maxSamples,
		unsigned int queryProbes, unsigned int hashingProbes, float tableAllocFraction, int myRank, int worldSize);

	/* Adds input vectors (in sparse format) to the hash table.
	Each vector is assigned ascending identification starting 0.
	For numInputEntries > 1, simply concatenate data vectors.

	@param numInputEntries: number of input vectors.
	@param dataIdx: non-zero indice of the sparse format.
	@param dataVal: non-zero values of the sparse format.
	@param dataMarker: marks the start index of each vector in dataIdx and dataVal. 
		Has an additional marker at the end to mark the (end+1) index.
	*/
	void add(int numInputEntries, int* dataIdx, float* dataVal, int* dataMarker, int dataOffset);

	/* Computes hashes for a partition of the set of query vectors.

	@param queryPartitionSize: the number of query vectors in the partition.
	@param numQueryPartitionHashes: the number of hashes that need to be computed for the partition.
	@param queryPartitionIndices: the indices of the non zero elements of each query vector.
	@param queryPartitionVals: the values of the non zero elements of each query vector.
	@param queryPartitionMarkers: the indices of the beginning and end of each query vector. 
		Has an additional marker at the end to mark the (end+1) index.
	@param queryHashes: an array to populate with the query hashes.
	*/
	void getQueryHash(int queryPartitionSize, int numQueryPartitionHashes, int* queryPartitionIndices, float* queryPartitionVals,
					  int* queryPartitionMarkers, unsigned int* queryHashes);

	/* Extractes the contents from the reservoirs of each hash table for some number of hashes.

	@param numQueryEntries: the number of queries to extract reservoirs for.
	@param segmentSize: the size of the block of memory for each query vector. 
		Equal to numQueryProbes * numTables * reservoirSize.
	@param queue: array to store the contents of the reservoirs.
	@param hashIndices: the hash indices of the query vectors, corresponding to a reservoir(s) in each table.
	*/
  	void extractReservoirs(int numQueryEntries, int segmentSize, unsigned int *queue, unsigned int *hashIndices);

	/* Print current parameter settings to the console.
	*/
	void showParams();

	/* Check the memory load of the hash table.
	*/
	void checkTableMemLoad();

	/* Prints contents of each hash table.
	*/
	void tableContents();

	/* Destructor. Frees memory allocations and OpenCL environments.
	*/
	~LSHReservoirSampler();
};

#endif