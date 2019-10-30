#include "omp.h"
#include "mpi.h"

#include "LSH.h"
#include "LSHReservoirSampler.h"
#include "CMS.h"
#include "dataset.h"
#include "flashControl.h"
#include "misc.h"
#include "evaluate.h"
#include "indexing.h"
#include "benchmarking.h"
#include "MatMul.h"
#include "FrequentItems.h"

#define TOPK_BENCHMARK

/*
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 */

void webspam()
{

/* ===============================================================
	MPI Initialization
*/
	int provided;
	MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
	int myRank, worldSize;
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	if (myRank == 0) {
		std::cout << "===============\n=== WEBSPAM ===\n===============\n" << std::endl;
	}

/* ===============================================================
	Data Structure Initialization
*/
	LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

	MPI_Barrier(MPI_COMM_WORLD);

	CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank, worldSize);

	MPI_Barrier(MPI_COMM_WORLD);

	LSHReservoirSampler *reservoir = new LSHReservoirSampler(lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION,
															 RANGE_ROW_U, NUM_DATA_VECTORS + NUM_QUERY_VECTORS, QUERY_PROBES, HASHING_PROBES, ALLOC_FRACTION, myRank, worldSize);

	MPI_Barrier(MPI_COMM_WORLD);

	flashControl *control = new flashControl(reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
											 DIMENSION, NUM_TABLES, QUERY_PROBES, RESERVOIR_SIZE);

/* ===============================================================
	Reading Data
*/
	std::cout << "\nReading Data Node " << myRank << "..." << std::endl;
	auto start = std::chrono::system_clock::now();

	control->allocateData(BASEFILE);
	MPI_Barrier(MPI_COMM_WORLD);	
	
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Data Read Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

/* ===============================================================
	Partitioning Query Between Nodes
*/
	
	control->allocateQuery(BASEFILE);

	MPI_Barrier(MPI_COMM_WORLD);

/* ===============================================================
	Adding Vectors
*/
	std::cout << "Adding Vectors Node " << myRank << "..." << std::endl;
	start = std::chrono::system_clock::now();
	control->add(NUM_BATCHES, BATCH_PRINT);
	end = std::chrono::system_clock::now();
	elapsed = end - start;
	std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

	// For debugging
	// control->printTables();

	MPI_Barrier(MPI_COMM_WORLD);

/* ===============================================================
	Hashing Query Vectors
*/
	std::cout << "Computing Query Hashes Node " << myRank << "..." << std::endl;
	start = std::chrono::system_clock::now();
	control->hashQuery();
	end = std::chrono::system_clock::now();
	elapsed = end - start;
	std::cout << "Query Hashes Computed Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

	MPI_Barrier(MPI_COMM_WORLD);

/* ===============================================================
	Extracting Reservoirs and Preforming Top-K selection
*/
	unsigned int *outputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
	start = std::chrono::system_clock::now();
	std::cout << "Extracting Top K (CMS) Node " << myRank << "..." << std::endl;
	//control->topKCMSAggregation(TOPK, outputs, 0);
	control->topKBruteForceAggretation(TOPK, outputs);
	end = std::chrono::system_clock::now();
	elapsed = end - start;
	std::cout << "Top K Extracted Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

	MPI_Barrier(MPI_COMM_WORLD);

/* ===============================================================
	De-allocating Data Structures in Memory
*/
	delete control;
	delete reservoir;
	delete lsh;
	delete cms;

/* ===============================================================
	MPI Closing
*/
	MPI_Finalize();

if (myRank == 0) {
/* ===============================================================
	Reading Groundtruths
*/
		unsigned int *gtruth_indice = new unsigned int[NUM_QUERY_VECTORS * AVAILABLE_TOPK];
		float *gtruth_dist = new float[NUM_QUERY_VECTORS * AVAILABLE_TOPK];
		std::cout << "Reading Groundtruth Node 0..." << std::endl;	
		start = std::chrono::system_clock::now();
		readGroundTruthInt(GTRUTHINDICE, NUM_QUERY_VECTORS, AVAILABLE_TOPK, gtruth_indice);
		readGroundTruthFloat(GTRUTHDIST, NUM_QUERY_VECTORS, AVAILABLE_TOPK, gtruth_dist);
		end = std::chrono::system_clock::now();
		elapsed = end - start;
		std::cout << "Groundtruth Read Node 0: " << elapsed.count() << " Seconds\n" << std::endl;

/* ===============================================================
	Similarity and Accuracy Calculations
*/
		int totalNumVectors = NUM_DATA_VECTORS + NUM_QUERY_VECTORS;
		int* sparseIndices = new int[totalNumVectors * DIMENSION];
		float* sparseVals = new float[totalNumVectors * DIMENSION];
		int* sparseMarkers = new int[totalNumVectors + 1];
			
		readSparse(BASEFILE, 0, totalNumVectors , sparseIndices, sparseVals, sparseMarkers, totalNumVectors * DIMENSION);

		const int nCnt = 10;
		int nList[nCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};
		const int gstdCnt = 8;
		float gstdVec[gstdCnt] = {0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.50};
		const int tstdCnt = 10;
		int tstdVec[tstdCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};

		std::cout << "\n\n================================\nTOP K CMS\n" << std::endl;

		similarityMetric(sparseIndices, sparseVals, sparseMarkers,
							sparseIndices, sparseVals, sparseMarkers, outputs, gtruth_dist,
							NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK, nList, nCnt);
		std::cout << "Similarity Metric Computed" << std::endl;
		// Commented out for testing purposes
		similarityOfData(gtruth_dist, NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK, nList, nCnt);
		std::cout << "Similarity of Data Computed" << std::endl;

		for (int i = 0; i < NUM_QUERY_VECTORS * TOPK; i++) {
			outputs[i] -= NUM_QUERY_VECTORS;
		}
		//Commented out for testing purposes
		//evaluate(outputs, NUM_QUERY_VECTORS, TOPK, gtruth_indice, gtruth_dist, AVAILABLE_TOPK, gstdVec, gstdCnt, tstdVec, tstdCnt, nList, nCnt);
		std::cout << "Evaluation Complete" << std::endl;

/* ===============================================================
	De-allocating Memory
*/
		delete[] gtruth_dist;
		delete[] gtruth_indice;
		delete[] sparseIndices;
		delete[] sparseVals;
		delete[] sparseMarkers;
	}
	delete[] outputs;
}

/*
 * KDD12
 * KDD12
 * KDD12
 * KDD12
 * KDD12
 * KDD12
 * KDD12
 * KDD12
 * KDD12
 * KDD12
 * KDD12
 * KDD12
 */

void kdd12()
{
/* ===============================================================
	MPI Initialization
*/
	int provided;
	MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
	int myRank, worldSize;
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	if (myRank == 0) {
		std::cout << "=============\n=== KDD12 ===\n=============\n" << std::endl;
	}

/* ===============================================================
	Data Structure Initialization
*/
	LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

	MPI_Barrier(MPI_COMM_WORLD);

	CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank, worldSize);

	MPI_Barrier(MPI_COMM_WORLD);

	LSHReservoirSampler *reservoir = new LSHReservoirSampler(lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION,
															 RANGE_ROW_U, NUM_DATA_VECTORS + NUM_QUERY_VECTORS, QUERY_PROBES, HASHING_PROBES, ALLOC_FRACTION, myRank, worldSize);

	MPI_Barrier(MPI_COMM_WORLD);

	flashControl *control = new flashControl(reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
											 DIMENSION, NUM_TABLES, QUERY_PROBES, RESERVOIR_SIZE);

/* ===============================================================
	Reading Data
*/
	std::cout << "\nReading Data Node " << myRank << "..." << std::endl;
	auto start = std::chrono::system_clock::now();

	control->allocateData(BASEFILE);
	MPI_Barrier(MPI_COMM_WORLD);	
	
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Data Read Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

/* ===============================================================
	Partitioning Query Between Nodes
*/
	
	control->allocateQuery(BASEFILE);

	MPI_Barrier(MPI_COMM_WORLD);

/* ===============================================================
	Adding Vectors
*/
	std::cout << "Adding Vectors Node " << myRank << "..." << std::endl;
	start = std::chrono::system_clock::now();
	control->add(NUM_BATCHES, BATCH_PRINT);
	end = std::chrono::system_clock::now();
	elapsed = end - start;
	std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

	MPI_Barrier(MPI_COMM_WORLD);

/* ===============================================================
	Hashing Query Vectors
*/
	std::cout << "Computing Query Hashes Node " << myRank << "..." << std::endl;
	start = std::chrono::system_clock::now();
	control->hashQuery();
	end = std::chrono::system_clock::now();
	elapsed = end - start;
	std::cout << "Query Hashes Computed Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

	MPI_Barrier(MPI_COMM_WORLD);

/* ===============================================================
	Extracting Reservoirs and Preforming Top-K selection
*/
	unsigned int *outputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
	start = std::chrono::system_clock::now();
	std::cout << "Extracting Top K (CMS) Node " << myRank << "..." << std::endl;
	//control->topKCMSAggregation(TOPK, outputs, 0);
	control->topKBruteForceAggretation(TOPK, outputs);
	end = std::chrono::system_clock::now();
	elapsed = end - start;
	std::cout << "Top K Extracted Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

	MPI_Barrier(MPI_COMM_WORLD);

/* ===============================================================
	De-allocating Data Structures in Memory
*/
	delete control;
	delete reservoir;
	delete lsh;
	delete cms;

/* ===============================================================
	MPI Closing
*/
	MPI_Finalize();

if (myRank == 0) {

/* ===============================================================
	Similarity and Accuracy Calculations
*/
		int totalNumVectors = NUM_DATA_VECTORS + NUM_QUERY_VECTORS;
		int* sparseIndices = new int[totalNumVectors * DIMENSION];
		float* sparseVals = new float[totalNumVectors * DIMENSION];
		int* sparseMarkers = new int[totalNumVectors + 1];
			
		readSparse(BASEFILE, 0, totalNumVectors , sparseIndices, sparseVals, sparseMarkers, totalNumVectors * DIMENSION);

		const int nCnt = 10;
		int nList[nCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};
		const int gstdCnt = 8;
		float gstdVec[gstdCnt] = {0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.50};
		const int tstdCnt = 10;
		int tstdVec[tstdCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};

		std::cout << "\n\n================================\nTOP K CMS\n" << std::endl;

		similarityMetric(sparseIndices, sparseVals, sparseMarkers,
							sparseIndices, sparseVals, sparseMarkers, outputs,
							NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK, nList, nCnt);
		std::cout << "Similarity Metric Computed" << std::endl;
		
/* ===============================================================
	De-allocating Memory
*/
		delete[] sparseIndices;
		delete[] sparseVals;
		delete[] sparseMarkers;
	}
	delete[] outputs;
}

/*
 * UNIT TESTING FUNCTION
 * UNIT TESTING FUNCTION
 * UNIT TESTING FUNCTION
 * UNIT TESTING FUNCTION
 * UNIT TESTING FUNCTION
 * UNIT TESTING FUNCTION
 * UNIT TESTING FUNCTION
 * UNIT TESTING FUNCTION
 * UNIT TESTING FUNCTION
 * UNIT TESTING FUNCTION
 * UNIT TESTING FUNCTION
 * UNIT TESTING FUNCTION
 */

void unitTesting()
{

/* ===============================================================
	MPI Initialization
*/
	int provided;
	MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
	int myRank, worldSize;
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

/* ===============================================================
	Data Structure Initialization
*/
	LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

	MPI_Barrier(MPI_COMM_WORLD);

	CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank, worldSize);

	MPI_Barrier(MPI_COMM_WORLD);

	LSHReservoirSampler *reservoir = new LSHReservoirSampler(lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION,
															 RANGE_ROW_U, NUM_DATA_VECTORS + NUM_QUERY_VECTORS, QUERY_PROBES, HASHING_PROBES, ALLOC_FRACTION, myRank, worldSize);

	MPI_Barrier(MPI_COMM_WORLD);

	flashControl *control = new flashControl(reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
											 DIMENSION, NUM_TABLES, QUERY_PROBES, RESERVOIR_SIZE);

/* ===============================================================
	Reading Data
*/
	if (myRank == 0) {
		std::cout << "\nReading Data Node 0..." << std::endl;
	}
	auto start = std::chrono::system_clock::now();

	control->allocateData(BASEFILE);
	MPI_Barrier(MPI_COMM_WORLD);	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed = end - start;
	if (myRank == 0) {
		std::cout << "Data Read Node 0: " << elapsed.count() << " Seconds\n" << std::endl;
	}

/* ===============================================================
	Partitioning Query Between Nodes
*/
	control->allocateQuery(BASEFILE);
	MPI_Barrier(MPI_COMM_WORLD);

/* ===============================================================
	Adding Vectors
*/
	std::cout << "Adding Vectors Node " << myRank << "..." << std::endl;
	start = std::chrono::system_clock::now();
	control->add(NUM_BATCHES, BATCH_PRINT);
	end = std::chrono::system_clock::now();
	elapsed = end - start;
	std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

	control->printTables();

	MPI_Barrier(MPI_COMM_WORLD);

/* ===============================================================
	Hashing Query Vectors
*/
	std::cout << "Computing Query Hashes Node " << myRank << "..." << std::endl;
	start = std::chrono::system_clock::now();
	control->hashQuery();
	end = std::chrono::system_clock::now();
	elapsed = end - start;
	std::cout << "Query Hashes Computed Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

	control->checkQueryHashes();

	MPI_Barrier(MPI_COMM_WORLD);


/* ===============================================================
	Extracting Reservoirs and Preforming Top-K selection
*/
	unsigned int *outputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
	start = std::chrono::system_clock::now();
	std::cout << "Extracting Top K (CMS) Node " << myRank << "..." << std::endl;
	control->topKCMSAggregation(TOPK, outputs, 0);
	// control->topKBruteForceAggretation(TOPK, outputs);
	end = std::chrono::system_clock::now();
	elapsed = end - start;
	std::cout << "Top K Extracted Node " << myRank << ": " << elapsed.count() << " Seconds\n" << std::endl;

	if (myRank == 0) {
		printf("Overall Top K\n");
		for (int k = 0; k < TOPK; k++) {
			printf("\tK %d: %d\n", k, outputs[k]);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

/* ===============================================================
	De-allocating Memory
*/
	delete[] outputs;
	delete control;
	delete reservoir;
	delete lsh;
	delete cms;

/* ===============================================================
	MPI Closing
*/
	MPI_Finalize();
}
