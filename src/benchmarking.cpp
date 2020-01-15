#include "mpi.h"
#include "omp.h"

#include "CMS.h"
#include "LSH.h"
#include "LSHReservoirSampler.h"
#include "benchmarking.h"
#include "dataset.h"
#include "flashControl.h"
#include "indexing.h"
#include "mathUtils.h"
#include <chrono>

#define TOPK_BENCHMARK

void showConfig(std::string dataset, int numVectors, int queries, int nodes,
                int tables, int rangePow, int reservoirSize, int hashes,
                int cmsHashes, int cmsBucketSize) {
  std::cout << "\n=================\n== " << dataset << "\n=================\n"
            << std::endl;

  printf("%d Vectors, %d Queries\n", numVectors, queries);

  printf(
      "Nodes: %d\nTables: %d\nRangePow: %d\nReservoir Size: %d\nHashes: %d\n",
      nodes, tables, rangePow, reservoirSize, hashes);

  printf("CMS Bucket Size: %d\nCMS Hashes: %d\n\n", cmsBucketSize, cmsHashes);
}

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

void webspam() {
  /* ===============================================================
MPI Initialization
*/
  int provided;
  MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
  int myRank, worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  if (myRank == 0) {
    showConfig("Webspam", NUM_DATA_VECTORS, NUM_QUERY_VECTORS, worldSize,
               NUM_TABLES, RANGE_POW, RESERVOIR_SIZE, NUM_HASHES, CMS_HASHES,
               CMS_BUCKET_SIZE);
  }

  /* ===============================================================
Data Structure Initialization
*/
  LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

  MPI_Barrier(MPI_COMM_WORLD);

  CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank,
                     worldSize);

  MPI_Barrier(MPI_COMM_WORLD);

  LSHReservoirSampler *reservoir = new LSHReservoirSampler(
      lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION, RANGE_ROW_U,
      NUM_DATA_VECTORS + NUM_QUERY_VECTORS, QUERY_PROBES, HASHING_PROBES,
      ALLOC_FRACTION, myRank, worldSize);

  MPI_Barrier(MPI_COMM_WORLD);

  flashControl *control = new flashControl(
      reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
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
  std::cout << "Data Read Node " << myRank << ": " << elapsed.count()
            << " Seconds\n"
            << std::endl;

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
  std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count()
            << " Seconds\n"
            << std::endl;

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
  std::cout << "Query Hashes Computed Node " << myRank << ": "
            << elapsed.count() << " Seconds\n"
            << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  /* ===============================================================
Extracting Reservoirs and Preforming Top-K selection
*/
  unsigned int *outputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
  start = std::chrono::system_clock::now();
  std::cout << "Extracting Top K (CMS) Node " << myRank << "..." << std::endl;
#ifdef CMS_AGGREGATION
  control->topKCMSAggregationTree(TOPK, outputs, 0);
#endif
#ifdef BF_AGGREGATION
  control->topKBruteForceAggretation(TOPK, outputs);
#endif
  end = std::chrono::system_clock::now();
  elapsed = end - start;
  std::cout << "Top K Extracted Node " << myRank << ": " << elapsed.count()
            << " Seconds\n"
            << std::endl;

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
    int *sparseIndices = new int[totalNumVectors * DIMENSION];
    float *sparseVals = new float[totalNumVectors * DIMENSION];
    int *sparseMarkers = new int[totalNumVectors + 1];

    readSparse(BASEFILE, 0, totalNumVectors, sparseIndices, sparseVals,
               sparseMarkers, totalNumVectors * DIMENSION);

    const int nCnt = 10;
    int nList[nCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};

    std::cout << "\n\n================================\nTOP K CMS\n"
              << std::endl;

    similarityMetric(sparseIndices, sparseVals, sparseMarkers, sparseIndices,
                     sparseVals, sparseMarkers, outputs, NUM_QUERY_VECTORS,
                     TOPK, AVAILABLE_TOPK, nList, nCnt);
    std::cout << "Similarity Metric Computed" << std::endl;
    std::cout << "Evaluation Complete" << std::endl;

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

void kdd12() {
  /* ===============================================================
MPI Initialization
*/
  int provided;
  MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
  int myRank, worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  if (myRank == 0) {
    showConfig("KDD12", NUM_DATA_VECTORS, NUM_QUERY_VECTORS, worldSize,
               NUM_TABLES, RANGE_POW, RESERVOIR_SIZE, NUM_HASHES, CMS_HASHES,
               CMS_BUCKET_SIZE);
  }

  /* ===============================================================
Data Structure Initialization
*/
  LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

  MPI_Barrier(MPI_COMM_WORLD);

  CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank,
                     worldSize);

  MPI_Barrier(MPI_COMM_WORLD);

  LSHReservoirSampler *reservoir = new LSHReservoirSampler(
      lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION, RANGE_ROW_U,
      NUM_DATA_VECTORS + NUM_QUERY_VECTORS, QUERY_PROBES, HASHING_PROBES,
      ALLOC_FRACTION, myRank, worldSize);

  MPI_Barrier(MPI_COMM_WORLD);

  flashControl *control = new flashControl(
      reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
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
  std::cout << "Data Read Node " << myRank << ": " << elapsed.count()
            << " Seconds\n"
            << std::endl;

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
  std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count()
            << " Seconds\n"
            << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  /* ===============================================================
Hashing Query Vectors
*/
  std::cout << "Computing Query Hashes Node " << myRank << "..." << std::endl;
  start = std::chrono::system_clock::now();
  control->hashQuery();
  end = std::chrono::system_clock::now();
  elapsed = end - start;
  std::cout << "Query Hashes Computed Node " << myRank << ": "
            << elapsed.count() << " Seconds\n"
            << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  /* ===============================================================
Extracting Reservoirs and Preforming Top-K selection
*/
  unsigned int *treeOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
  start = std::chrono::system_clock::now();
  std::cout << "Extracting Top K (TREE) Node " << myRank << "..." << std::endl;
  control->topKCMSAggregationTree(TOPK, treeOutputs, 0);
  end = std::chrono::system_clock::now();
  elapsed = end - start;
  std::cout << "Top K (TREE) Extracted Node " << myRank << ": "
            << elapsed.count() << " Seconds\n"
            << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  // ==============================================

  unsigned int *linearOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
  start = std::chrono::system_clock::now();
  std::cout << "Extracting Top K (LINEAR) Node " << myRank << "..."
            << std::endl;
  control->topKCMSAggregationLinear(TOPK, linearOutputs, 0);
  end = std::chrono::system_clock::now();
  elapsed = end - start;
  std::cout << "Top K (LINEAR) Extracted Node " << myRank << ": "
            << elapsed.count() << " Seconds\n"
            << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  // ==============================================

  unsigned int *bruteforceOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
  start = std::chrono::system_clock::now();
  std::cout << "Extracting Top K (BRUTEFORCE) Node " << myRank << "..."
            << std::endl;
  control->topKBruteForceAggretation(TOPK, bruteforceOutputs);
  end = std::chrono::system_clock::now();
  elapsed = end - start;
  std::cout << "Top K (BRUTEFORCE) Extracted Node " << myRank << ": "
            << elapsed.count() << " Seconds\n"
            << std::endl;

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
    int *sparseIndices = new int[totalNumVectors * DIMENSION];
    float *sparseVals = new float[totalNumVectors * DIMENSION];
    int *sparseMarkers = new int[totalNumVectors + 1];

    readSparse(BASEFILE, 0, totalNumVectors, sparseIndices, sparseVals,
               sparseMarkers, totalNumVectors * DIMENSION);

    const int nCnt = 10;
    int nList[nCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};

    std::cout << "\n\n================================\nTOP K TREE\n"
              << std::endl;

    similarityMetric(sparseIndices, sparseVals, sparseMarkers, sparseIndices,
                     sparseVals, sparseMarkers, treeOutputs, NUM_QUERY_VECTORS,
                     TOPK, AVAILABLE_TOPK, nList, nCnt);

    std::cout << "\n\n================================\nTOP K LINEAR\n"
              << std::endl;

    similarityMetric(sparseIndices, sparseVals, sparseMarkers, sparseIndices,
                     sparseVals, sparseMarkers, linearOutputs,
                     NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK, nList, nCnt);

    std::cout << "\n\n================================\nTOP K BRUTEFORCE\n"
              << std::endl;

    similarityMetric(sparseIndices, sparseVals, sparseMarkers, sparseIndices,
                     sparseVals, sparseMarkers, bruteforceOutputs,
                     NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK, nList, nCnt);

    std::cout << "Similarity Metric Computed" << std::endl;

    /* ===============================================================
De-allocating Memory
*/
    delete[] sparseIndices;
    delete[] sparseVals;
    delete[] sparseMarkers;
  }
  delete[] treeOutputs;
  delete[] linearOutputs;
  delete[] bruteforceOutputs;
}

void kdd12FileOutput() {
  /* ===============================================================
MPI Initialization
*/
  int provided;
  MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
  int myRank, worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  if (myRank == 0) {
    showConfig("KDD12", NUM_DATA_VECTORS, NUM_QUERY_VECTORS, worldSize,
               NUM_TABLES, RANGE_POW, RESERVOIR_SIZE, NUM_HASHES, CMS_HASHES,
               CMS_BUCKET_SIZE);
  }

  /* ===============================================================
Data Structure Initialization
*/
  LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

  MPI_Barrier(MPI_COMM_WORLD);

  CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank,
                     worldSize);

  MPI_Barrier(MPI_COMM_WORLD);

  LSHReservoirSampler *reservoir = new LSHReservoirSampler(
      lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION, RANGE_ROW_U,
      NUM_DATA_VECTORS + NUM_QUERY_VECTORS, QUERY_PROBES, HASHING_PROBES,
      ALLOC_FRACTION, myRank, worldSize);

  MPI_Barrier(MPI_COMM_WORLD);

  flashControl *control = new flashControl(
      reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
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
  std::cout << "Data Read Node " << myRank << ": " << elapsed.count()
            << " Seconds\n"
            << std::endl;

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
  std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count()
            << " Seconds\n"
            << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  /* ===============================================================
Hashing Query Vectors
*/
  std::cout << "Computing Query Hashes Node " << myRank << "..." << std::endl;
  start = std::chrono::system_clock::now();
  control->hashQuery();
  end = std::chrono::system_clock::now();
  elapsed = end - start;
  std::cout << "Query Hashes Computed Node " << myRank << ": "
            << elapsed.count() << " Seconds\n"
            << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  /* ===============================================================
Extracting Reservoirs and Preforming Top-K selection
*/
  unsigned int *treeOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
  start = std::chrono::system_clock::now();
  std::cout << "Extracting Top K (TREE) Node " << myRank << "..." << std::endl;
  control->topKCMSAggregationTree(TOPK, treeOutputs, 0);
  end = std::chrono::system_clock::now();
  elapsed = end - start;
  std::cout << "Top K (TREE) Extracted Node " << myRank << ": "
            << elapsed.count() << " Seconds\n"
            << std::endl;

  std::string filenameTree("Tree-Nodes-");
  filenameTree.append(std::to_string(worldSize));
  if (myRank == 0) {
    writeTopK(filenameTree, NUM_QUERY_VECTORS, TOPK, treeOutputs);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // ==============================================

  unsigned int *linearOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
  start = std::chrono::system_clock::now();
  std::cout << "Extracting Top K (LINEAR) Node " << myRank << "..."
            << std::endl;
  control->topKCMSAggregationLinear(TOPK, linearOutputs, 0);
  end = std::chrono::system_clock::now();
  elapsed = end - start;
  std::cout << "Top K (LINEAR) Extracted Node " << myRank << ": "
            << elapsed.count() << " Seconds\n"
            << std::endl;

  std::string filenameLinear("Linear-Nodes-");
  filenameLinear.append(std::to_string(worldSize));
  if (myRank == 0) {
    writeTopK(filenameLinear, NUM_QUERY_VECTORS, TOPK, linearOutputs);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // ==============================================

  unsigned int *bruteforceOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
  start = std::chrono::system_clock::now();
  std::cout << "Extracting Top K (BRUTEFORCE) Node " << myRank << "..."
            << std::endl;
  control->topKBruteForceAggretation(TOPK, bruteforceOutputs);
  end = std::chrono::system_clock::now();
  elapsed = end - start;
  std::cout << "Top K (BRUTEFORCE) Extracted Node " << myRank << ": "
            << elapsed.count() << " Seconds\n"
            << std::endl;

  std::string filenameBruteforce("Bruteforce-Nodes-");
  filenameBruteforce.append(std::to_string(worldSize));
  if (myRank == 0) {
    writeTopK(filenameBruteforce, NUM_QUERY_VECTORS, TOPK, bruteforceOutputs);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /* ===============================================================
De-allocating Data Structures in Memory
*/
  delete control;
  delete reservoir;
  delete lsh;
  delete cms;
  delete[] treeOutputs;
  delete[] linearOutputs;
  delete[] bruteforceOutputs;
  /* ===============================================================
MPI Closing
*/
  MPI_Finalize();
}

/*
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 */

void wikiDump() {
  /* ===============================================================
MPI Initialization
*/
  int provided;
  MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
  int myRank, worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  if (myRank == 0) {
    showConfig("WikiDump", NUM_DATA_VECTORS, NUM_QUERY_VECTORS, worldSize,
               NUM_TABLES, RANGE_POW, RESERVOIR_SIZE, NUM_HASHES, CMS_HASHES,
               CMS_BUCKET_SIZE);
  }

  /* ===============================================================
Data Structure Initialization
*/
  LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

  MPI_Barrier(MPI_COMM_WORLD);

  CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank,
                     worldSize);

  MPI_Barrier(MPI_COMM_WORLD);

  LSHReservoirSampler *reservoir = new LSHReservoirSampler(
      lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION, RANGE_ROW_U,
      NUM_DATA_VECTORS + NUM_QUERY_VECTORS, QUERY_PROBES, HASHING_PROBES,
      ALLOC_FRACTION, myRank, worldSize);

  MPI_Barrier(MPI_COMM_WORLD);

  flashControl *control = new flashControl(
      reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
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
  std::cout << "Data Read Node " << myRank << ": " << elapsed.count()
            << " Seconds\n"
            << std::endl;

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
  std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count()
            << " Seconds\n"
            << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  /* ===============================================================
Hashing Query Vectors
*/
  std::cout << "Computing Query Hashes Node " << myRank << "..." << std::endl;
  start = std::chrono::system_clock::now();
  control->hashQuery();
  end = std::chrono::system_clock::now();
  elapsed = end - start;
  std::cout << "Query Hashes Computed Node " << myRank << ": "
            << elapsed.count() << " Seconds\n"
            << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  /* ===============================================================
Extracting Reservoirs and Preforming Top-K selection
*/
  unsigned int *treeOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
  start = std::chrono::system_clock::now();
  std::cout << "Extracting Top K (TREE) Node " << myRank << "..." << std::endl;
  control->topKCMSAggregationTree(TOPK, treeOutputs, 0);
  end = std::chrono::system_clock::now();
  elapsed = end - start;
  std::cout << "Top K (TREE) Extracted Node " << myRank << ": "
            << elapsed.count() << " Seconds\n"
            << std::endl;

  std::string filenameTree("Tree-Nodes-");
  filenameTree.append(std::to_string(worldSize));
  if (myRank == 0) {
    writeTopK(filenameTree, NUM_QUERY_VECTORS, TOPK, treeOutputs);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /* ===============================================================
De-allocating Data Structures in Memory
*/
  delete control;
  delete reservoir;
  delete lsh;
  delete cms;
  delete[] treeOutputs;

  /* ===============================================================
MPI Closing
*/
  MPI_Finalize();
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

void unitTesting() {
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

  CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank,
                     worldSize);

  MPI_Barrier(MPI_COMM_WORLD);

  LSHReservoirSampler *reservoir = new LSHReservoirSampler(
      lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION, RANGE_ROW_U,
      NUM_DATA_VECTORS + NUM_QUERY_VECTORS, QUERY_PROBES, HASHING_PROBES,
      ALLOC_FRACTION, myRank, worldSize);

  MPI_Barrier(MPI_COMM_WORLD);

  flashControl *control = new flashControl(
      reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
      DIMENSION, NUM_TABLES, QUERY_PROBES, RESERVOIR_SIZE);

  /* ===============================================================
Reading Data
*/
  if (myRank == 0) {
    std::cout << "\nReading Data Node 0..." << std::endl;
  }
  auto start = std::chrono::system_clock::now();

  control->allocateData(BASEFILE);
  MPI_Barrier(MPI_COMM_WORLD);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  if (myRank == 0) {
    std::cout << "Data Read Node 0: " << elapsed.count() << " Seconds\n"
              << std::endl;
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
  std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count()
            << " Seconds\n"
            << std::endl;

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
  std::cout << "Query Hashes Computed Node " << myRank << ": "
            << elapsed.count() << " Seconds\n"
            << std::endl;

  control->checkQueryHashes();

  MPI_Barrier(MPI_COMM_WORLD);

  /* ===============================================================
Extracting Reservoirs and Preforming Top-K selection
*/
  unsigned int *outputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
  start = std::chrono::system_clock::now();
  std::cout << "Extracting Top K (CMS) Node " << myRank << "..." << std::endl;
  control->topKCMSAggregationTree(TOPK, outputs, 0);
  // control->topKBruteForceAggretation(TOPK, outputs);
  end = std::chrono::system_clock::now();
  elapsed = end - start;
  std::cout << "Top K Extracted Node " << myRank << ": " << elapsed.count()
            << " Seconds\n"
            << std::endl;

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

void evaluateResults(std::string resultFile) {

  int totalNumVectors = NUM_DATA_VECTORS + NUM_QUERY_VECTORS;
  unsigned int *outputs = new unsigned int[NUM_QUERY_VECTORS * TOPK];
  readTopK(resultFile, NUM_QUERY_VECTORS, TOPK, outputs);

  int *sparseIndices = new int[totalNumVectors * DIMENSION];
  float *sparseVals = new float[totalNumVectors * DIMENSION];
  int *sparseMarkers = new int[totalNumVectors + 1];

  readSparse(BASEFILE, 0, totalNumVectors, sparseIndices, sparseVals,
             sparseMarkers, totalNumVectors * DIMENSION);

  const int nCnt = 10;
  int nList[nCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};

  std::cout << "\n\n================================\nTOP K TREE\n"
            << std::endl;

  similarityMetric(sparseIndices, sparseVals, sparseMarkers, sparseIndices,
                   sparseVals, sparseMarkers, outputs, NUM_QUERY_VECTORS, TOPK,
                   AVAILABLE_TOPK, nList, nCnt);
}