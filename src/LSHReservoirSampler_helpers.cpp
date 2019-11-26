#include "LSHReservoirSampler.h"
#include "omp.h"

void LSHReservoirSampler::reservoirSampling(unsigned int *allprobsHash, unsigned int *allprobsIdx,
	unsigned int *storelog, int numProbePerTb) {

	unsigned int counter, allocIdx, reservoirRandNum, TB, hashIdx, inputIdx, ct, reservoir_full, location;

#pragma omp parallel for private(TB, hashIdx, inputIdx, ct, allocIdx, counter, reservoir_full, reservoirRandNum, location)
	for (int probeIdx = 0; probeIdx < numProbePerTb; probeIdx++) {
		for (unsigned int tb = 0; tb < _numTables; tb++) {

			TB = numProbePerTb * tb;

			hashIdx = allprobsHash[allprobsHashSimpleIdx(numProbePerTb, tb, probeIdx)];
			inputIdx = allprobsIdx[allprobsHashSimpleIdx(numProbePerTb, tb, probeIdx)];
			ct = 0;

			/* Allocate the reservoir if non-existent. */
			omp_set_lock(_tablePointersLock + tablePointersIdx(_numReservoirsHashed, hashIdx, tb, _sechash_a, _sechash_b));
			allocIdx = _tablePointers[tablePointersIdx(_numReservoirsHashed, hashIdx, tb, _sechash_a, _sechash_b)];
			if (allocIdx == TABLENULL) {
				allocIdx = _tableMemAllocator[tableMemAllocatorIdx(tb)];
				_tableMemAllocator[tableMemAllocatorIdx(tb)] ++;
				_tablePointers[tablePointersIdx(_numReservoirsHashed, hashIdx, tb, _sechash_a, _sechash_b)] = allocIdx;
			}
			omp_unset_lock(_tablePointersLock + tablePointersIdx(_numReservoirsHashed, hashIdx, tb, _sechash_a, _sechash_b));

			// ATOMIC: Obtain the counter, and increment the counter. (Counter initialized to 0 automatically).
			// Counter counts from 0 to currentCount-1.
			omp_set_lock(_tableCountersLock + tableCountersLockIdx(tb, allocIdx, _aggNumReservoirs));

			counter = _tableMem[tableMemCtIdx(tb, allocIdx, _aggNumReservoirs)]; // Potentially overflowable.
			_tableMem[tableMemCtIdx(tb, allocIdx, _aggNumReservoirs)] ++;
			omp_unset_lock(_tableCountersLock + tableCountersLockIdx(tb, allocIdx, _aggNumReservoirs));

			// The counter here is the old counter. Current count is already counter + 1.
			// If current count is larger than _reservoirSize, current item needs to be sampled.
			//reservoir_full = (counter + 1) > _reservoirSize;

			reservoirRandNum = _global_rand[std::min((unsigned int)(_maxReservoirRand-1), counter)]; // Overflow prevention.

			if ((counter + 1) > _reservoirSize) { // Reservoir full.
				location = reservoirRandNum;
			}
			else {
				location = counter;
			}

			//location = reservoir_full * (reservoirRandNum)+(1 - reservoir_full) * counter;

			storelog[storelogIdIdx(numProbePerTb, probeIdx, tb)] = inputIdx;
			storelog[storelogCounterIdx(numProbePerTb, probeIdx, tb)] = counter;
			storelog[storelogLocationIdx(numProbePerTb, probeIdx, tb)] = location;
			storelog[storelogHashIdxIdx(numProbePerTb, probeIdx, tb)] = hashIdx;

		}
	}
}

void LSHReservoirSampler::addTable(unsigned int *storelog, int numProbePerTb, int dataOffset) {

	unsigned int id, hashIdx, allocIdx;
	unsigned locCapped;
//#pragma omp parallel for private(allocIdx, id, hashIdx, locCapped)
	for (int probeIdx = 0; probeIdx < numProbePerTb; probeIdx++) {
		for (unsigned int tb = 0; tb < _numTables; tb++) {

			id = storelog[storelogIdIdx(numProbePerTb, probeIdx, tb)];
			hashIdx = storelog[storelogHashIdxIdx(numProbePerTb, probeIdx, tb)];
			allocIdx = _tablePointers[tablePointersIdx(_numReservoirsHashed, hashIdx, tb, _sechash_a, _sechash_b)];
			// If item_i spills out of the reservoir, it is capped to the dummy location at _reservoirSize.
			locCapped = storelog[storelogLocationIdx(numProbePerTb, probeIdx, tb)];

			if (locCapped < _reservoirSize) {
				_tableMem[tableMemResIdx(tb, allocIdx, _aggNumReservoirs) + locCapped] = id + _sequentialIDCounter_kernel + dataOffset;
			}
		}
	}
}