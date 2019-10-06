#ifndef _FLASH_CONTROL_H
#define _FLASH_CONTROL_H

#include <iostream>
#include "CMS.h"
#include <string>
#include "mpi.h"
#include "LSHReservoirSampler.h"
#include "dataset.h"

class flashControl {

private:
    int _myRank, _worldSize;

    LSHReservoirSampler* _myReservoir;

    CMS* _mySketch;

    // Reservoir Params
    int _numTables, _numQueryProbes, _reservoirSize;

    int _numDataVectors, _numQueryVectors; // Total number of data and query vectors across all nodes

    int* _dataVectorCts; // Number of data vectors allocated to each node
    int* _dataVectorOffsets; // Offset, in number of vectors, for the range of vectors of each node
    int* _dataCts; // Total length of all data vectors allocated to each node
    int* _dataOffsets; // Offset, in total length of data vectors, of the data range of each node

    int* _queryVectorCts; // Number of query vectors allocated to each node
    int* _queryVectorOffsets; // Offset, in number of vectors, for the range of vectors of each node
    int* _queryCts; // Total length of all query vectors allocated to each node
    int* _queryOffsets; // Offset, in total length of query vectors, of the query range of each node

    

    // For storing the partition of the data allocated to each node
    int _myDataVectorsCt; // Number of data vectors allocated to a specific node
    int _myDataVectorsLen; // Combined length of all of the data vectors allocated to a specific node
    int _myDataVectorsOffset; // Offset of data vectory array for a specific node
    int _myDataOffset;
    int* _myDataIndices; // Locations of non-zeros within a node's partition of data vectors
    float* _myDataVals; // Values of non-zeros within a node's partition of data vectors
    int* _myDataMarkers; // Start and end indexes of data vectors within a node's partition of data vectors

    // For storing the partition of the query vectors allocated to each node
    int _myQueryVectorsCt; // Number of query vectors allocated to a specific node
    int _myQueryVectorsLen; // Combined length of all of the query vectors allocated to a specific node
    int* _myQueryIndices; // Location of non-zeros within a node's partition of query vectors
    float* _myQueryVals; // Values of non-zeros within a node's partition of query vectors
    int* _myQueryMarkers; // Start and end indexes of query vectors within a node's partition of query vectors

    int _myHashCt; // Number of hashes computed by specific node
    int* _hashCts;  // Number of hashes computed by each node
    int* _hashOffsets; // Offsets of hash array for each node

    unsigned int* _allQueryHashes; // Combined hashes from all nodes

public:

    // For storing full dataset, only used in node 0
    int* _sparseIndices; // Locations of non-zeros
    float* _sparseVals; // Vals of non-zeros
    int* _sparseMarkers; // Start and end indexes of data vectors
    
    /* Constructor. 
        Initializes a FLASH Controller object that manages communication and data partitions between nodes.
        Determines how the many data vectors will be sent to each node and how many query vectors each node will hash.

    @param reservoir: a LSHReservoirSampler object.
    @param cms: a count min sketch/topKAPI obect.
    @param myRank: the rank of node calling the constructor.
    @param worldSize: the total number of nodes.
    @param numDataVectors: the total number of dataVectors that will be added across all nodes.
    @param numQueryVectors: the total number of queryVectors that will used across all nodes.
    @param numTables: the number of tables in the instance of LSHReservoirSampler.
    @param numQueryProbes: the number of probes used for each query vector.
    @param reservoirSize: the size of each reservoir in the instance of LSHReservoirSampler.
    */
    flashControl(LSHReservoirSampler* reservoir, CMS* cms, int myRank, int worldSize, int numDataVectors, 
                 int numQueryVectors, int numTables, int numQueryProbes, int reservoirSize);

    /* Reads the data from a file, determines the indices of the partitions of data and query vectors
       that each node will be responsible for.

       @param filename: the file to read the data from.
       @param dataSetSize: the total number of vectors to read.
       @param dimension: max length of each vector, used for allocating memory.
    */
    void readData(std::string filename, int dataSetSize, int dimension);

    // Allocates memory in each node and sends each node its partition of the set of data vectors.
    void allocateData();

    // Allocates memory in each node and sends each node its partition of the set of query vectors for hashing.
    void allocateQuery();

    /* Adds a nodes set of data vectors to its LSHReservoirSampler object.

    @param numBatches: the number of batches to break the data into when adding.
    @param batchPrint: after each set of this number of batches the function will print the memory usage 
        of each hash table.
    */
    void add(int numBatches, int batchPrint);

    // Computes the hashes of each partition of the query vectors in each node, and then combines each 
    // partition of hashes into a single set of hashes in every node.
    void hashQuery();

    // For testing
    // For use with single node to establish benchmark for comparison with TopKAPI top-k selection.
    void extractReservoirs(int topK, unsigned int* outputs);

    /* Extracts reservoirs from each node's hash tables, stores frequency counts for each node 
        in  a CMS object, aggregates CMS objects in node 0, and preforms top-k selection there.

    @param topK: the number of top elements to select.
    @param outputs: an array to store the top-k selected.
    @param threshold: used for extracting heavy hitters in topKAPI
    */
    void extractReservoirsCMS(int topK, unsigned int* outputs, int threshold);

    // For debugging: shows the partitions of the data and query allocated to a specific node.
    void showPartitions();

    void localTopK(int topK, unsigned int* outputs, int threshold);

    void printData();

    void checkDataTransfer();

    // Destructor
    ~flashControl();

};

#endif
