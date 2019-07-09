#include "LSH.h"

#ifndef INT_MAX
#define INT_MAX 0xffffffff
#endif

// The range of hashes returned by getRandDoubleHash is 2^_lognumhash = numhash.
unsigned int LSH::getRandDoubleHash(int binid, int count) {
	unsigned int tohash = ((binid + 1) << 10) + count;
	return ((unsigned int)_randHash[0] * tohash << 3) >> (32 - _lognumhash); // _lognumhash needs to be ceiled.
}

void LSH::optimalMinHash(unsigned int *hashArray, unsigned int *nonZeros, int sizenonzeros) {
	/* This function computes the minhash and perform densification. */
	unsigned int *hashes = new unsigned int[_numhashes];

	unsigned int range = 1 << _rangePow;
	// binsize is the number of times the range is larger than the total number of hashes we need.
	unsigned int binsize = ceil(range / _numhashes);

	for (size_t i = 0; i < _numhashes; i++)
	{
		hashes[i] = INT_MAX;
	}
	
	for (size_t i = 0; i < sizenonzeros; i++)
	{
		unsigned int h = nonZeros[i];
		h *= _randa;
		h ^= h >> 13;
		h *= 0x85ebca6b;
		unsigned int curhash = ((unsigned int)(((unsigned int)h*nonZeros[i]) << 5) >> (32 - _rangePow));
		unsigned int binid = std::min((unsigned int) floor(curhash / binsize), (unsigned int)(_numhashes - 1));
		if (hashes[binid] > curhash)
			hashes[binid] = curhash;
	}
	/* Densification of the hash. */
	for (size_t i = 0; i < _numhashes; i++)
	{
		unsigned int next = hashes[i];
		if (next != INT_MAX)
		{
			hashArray[i] = hashes[i];
			continue;
		}
		unsigned int count = 0;
		while (next == INT_MAX)
		{
			count++;
			unsigned int index = std::min(
				(unsigned)getRandDoubleHash((unsigned int)i, count),
				(unsigned)_numhashes);
			next = hashes[index]; // Kills GPU.
			if (count > 100) // Densification failure.
				break;
		}
		hashArray[i] = next;
	}
	delete[] hashes;
}