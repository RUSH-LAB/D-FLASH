#include <cmath>

#include "LSHReservoirSampler.h"
#include "dataset.h"
#include "misc.h"
#include "evaluate.h"
#include "indexing.h"
#include "omp.h"
#include "MatMul.h"
#include "benchmarking.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include "FrequentItems.h"

#include <iostream>

void checkFileIO() {
	int provided;
	MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
	int myRank, worldSize;
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	int* sparseMarkers = new int[11];
	int* sparseIndices = new int[10 * 5000];
	float* sparseVals = new float[10* 5000];

	for (int n = 0; n < worldSize; n++) {
		if (myRank == n) {
			printf("Reading File Node %d\n", n);
			readSparse(BASEFILE, 10 * myRank, 10, sparseIndices, sparseVals, sparseMarkers, 10 * 5000);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Finalize();
}

int main() {

#ifdef UNIT_TESTING	
	unitTesting();
#endif
#ifdef WEBSPAM
	// webspamTest();
#endif
	checkFileIO();
	return 0;
}
