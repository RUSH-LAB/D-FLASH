#include "flashControl.h"

void flashControl::allocateData(std::string filename) {
    _myDataIndices = new int[_myDataVectorsCt * _dimension];
    _myDataVals = new float[_myDataVectorsCt * _dimension];
    _myDataMarkers = new int[_myDataVectorsCt + 1];

    readSparse(filename, _myDataVectorsOffset, _myDataVectorsCt, _myDataIndices, _myDataVals, _myDataMarkers, _myDataVectorsCt * _dimension);
}

void flashControl::allocateQuery(std::string filename) {

    if (_myRank == 0) {
        _queryIndices = new int[(unsigned)(_numQueryVectors * _dimension)];
        _queryVals = new float[(unsigned)(_numDataVectors * _dimension)];
        _queryMarkers = new int[(unsigned)(_numQueryVectors + 1)];
        readSparse(filename, 0, (unsigned) _numQueryVectors, _queryIndices, _queryVals, _queryMarkers, (unsigned)(_numQueryVectors * _dimension));

        for (int n = 0; n < _worldSize; n++) {
            _queryOffsets[n] = _queryMarkers[_queryVectorOffsets[n]];
            _queryCts[n] = _queryMarkers[_queryVectorOffsets[n] + _queryVectorCts[n]] - _queryOffsets[n];
        }
    }
    
    MPI_Bcast(_queryOffsets, _worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(_queryCts, _worldSize, MPI_INT, 0, MPI_COMM_WORLD);

    _myQueryVectorsLen = _queryCts[_myRank];

    _myQueryIndices = new int[_myQueryVectorsLen];
    _myQueryVals = new float[_myQueryVectorsLen];
    _myQueryMarkers = new int[_myQueryVectorsCt + 1];

    int* tempQueryMarkerCts = new int[_worldSize];
    for (int n = 0; n < _worldSize; n++) {
        tempQueryMarkerCts[n] = _queryVectorCts[n] + 1; // To account for extra element at the end of each marker array
    }

    MPI_Scatterv(_queryIndices, _queryCts, _queryOffsets, MPI_INT, _myQueryIndices, _myQueryVectorsLen, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(_queryVals, _queryCts, _queryOffsets, MPI_FLOAT, _myQueryVals, _myQueryVectorsLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(_queryMarkers, tempQueryMarkerCts, _queryVectorOffsets, MPI_INT, _myQueryMarkers, _myQueryVectorsCt + 1, MPI_INT, 0, MPI_COMM_WORLD);

    int myQueryOffset = _queryOffsets[_myRank];
    for (int i = 0; i < _myQueryVectorsCt + 1; i++) {
        _myQueryMarkers[i] -= myQueryOffset;
    }
    delete[] tempQueryMarkerCts;
}

void flashControl::add(int numBatches, int batchPrint) {
    int batchSize = _myDataVectorsCt / numBatches;
    for (int batch = 0; batch < numBatches; batch++) {
        _myReservoir->add(batchSize, _myDataIndices, _myDataVals, _myDataMarkers + batch * batchSize, _myDataVectorsOffset);
        if (batch % batchPrint == 0) {
            _myReservoir->checkTableMemLoad();
        }
    }
}

void flashControl::hashQuery() {

    unsigned int* myPartitionHashes = new unsigned int[_myHashCt];

    _myReservoir->getQueryHash(_myQueryVectorsCt, _myHashCt, _myQueryIndices, _myQueryVals, _myQueryMarkers, myPartitionHashes);

    unsigned int* queryHashBuffer = new unsigned int[_numQueryVectors * _numQueryProbes * _numTables];

    MPI_Allgatherv(myPartitionHashes, _myHashCt, MPI_UNSIGNED, queryHashBuffer, _hashCts, _hashOffsets, MPI_UNSIGNED, MPI_COMM_WORLD);

    unsigned int len;

    unsigned int* old;
    unsigned int* fin;

#pragma omp parallel for default(none) shared(queryHashBuffer, _allQueryHashes, _hashOffsets, _numQueryProbes, _numTables) private(len, old, fin)
    for (int partition = 0; partition < _worldSize; partition++) {
        len = _queryVectorCts[partition] * _numQueryProbes;
        for (int tb = 0; tb < _numTables; tb++) {
            old = queryHashBuffer + _hashOffsets[partition] + tb * len;
            fin = _allQueryHashes + tb * _numQueryVectors * _numQueryProbes + (_hashOffsets[partition] / _numTables);
            for (int l = 0; l < len; l++) {
                fin[l] = old[l];
            }
        }
    }

    delete[] queryHashBuffer;
    delete[] myPartitionHashes;
}

void flashControl::topKBruteForceAggretation(int topK, unsigned int* outputs) {
    int segmentSize = _numTables * _numQueryProbes * _reservoirSize;
    unsigned int* allReservoirsExtracted = new unsigned int[segmentSize * _numQueryVectors];
    _myReservoir->extractReservoirs(_numQueryVectors, segmentSize, allReservoirsExtracted, _allQueryHashes);

    unsigned int* allReservoirsAllNodes;
    if (_myRank == 0) {
        allReservoirsAllNodes = new unsigned int[segmentSize * _numQueryVectors * _worldSize];
    }
    MPI_Gather(allReservoirsExtracted, segmentSize * _numQueryVectors, MPI_UNSIGNED, allReservoirsAllNodes, segmentSize * _numQueryVectors, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    if (_myRank == 0) {
        unsigned int* allReservoirsAllNodesOrdered = new unsigned int[segmentSize * _numQueryVectors * _worldSize];
        
        int queryBlockSize = _worldSize * segmentSize;
        unsigned int* old;
        unsigned int* final;

#pragma omp parallel for default(none) shared(allReservoirsAllNodes, allReservoirsAllNodesOrdered, queryBlockSize, segmentSize, _numQueryVectors) private(old, final)
        for(int v = 0; v < _numQueryVectors; v++) {
            for (int n = 0; n < _worldSize; n++) {
                old = allReservoirsAllNodes + v * segmentSize + n * (_numQueryVectors * segmentSize);
                final = allReservoirsAllNodesOrdered + v * queryBlockSize + n * segmentSize;
                for (int i = 0; i < segmentSize; i++) {
                    final[i] = old[i];
                }
            }
        }

        delete[] allReservoirsAllNodes;

#pragma omp parallel for default(none) shared(allReservoirsAllNodesOrdered, queryBlockSize)
        for (int v = 0; v < _numQueryVectors; v++) {
            std::sort(allReservoirsAllNodesOrdered + v * queryBlockSize, allReservoirsAllNodesOrdered + (v + 1) * queryBlockSize);
        }

        VectorFrequency* vectorCnts = new VectorFrequency[segmentSize * _numQueryVectors * _worldSize];

#pragma omp parallel for default(none) shared(allReservoirsAllNodesOrdered, vectorCnts, queryBlockSize, outputs, topK)
        for (int v = 0; v < _numQueryVectors; v++) {
            int uniqueVectors = 0;
            unsigned int current = allReservoirsAllNodesOrdered[0];
            int count = 1;
            for (int i = 1; i < queryBlockSize; i++) {
                if (allReservoirsAllNodesOrdered[i + v * queryBlockSize] == current) {
                    count++;
                } else {
                    vectorCnts[uniqueVectors + v * queryBlockSize].vector = current;
                    vectorCnts[uniqueVectors + v * queryBlockSize].count = count;
                    current = allReservoirsAllNodesOrdered[i + v * queryBlockSize];
                    count = 1;
                    uniqueVectors++;
                }
            }
            vectorCnts[uniqueVectors + v * queryBlockSize].vector = current;
            vectorCnts[uniqueVectors + v * queryBlockSize].count = count;
			uniqueVectors++;
            for (; uniqueVectors < queryBlockSize; uniqueVectors++) {
                vectorCnts[uniqueVectors + v * queryBlockSize].count = -1;
            }
            std::sort(vectorCnts + v * queryBlockSize, vectorCnts + (v + 1) * queryBlockSize, [&vectorCnts](VectorFrequency a, VectorFrequency b){return a.count > b.count;});
            int s = 0;
            if (vectorCnts[queryBlockSize * v].vector == 0) s++;
            for (int k = 0; k < topK; k++) {
                outputs[k + topK * v] = vectorCnts[s + k + v * queryBlockSize].vector;
            }
        }
    }
}

void flashControl::topKCMSAggregation(int topK, unsigned int* outputs, int threshold) {
    int segmentSize = _numTables * _numQueryProbes * _reservoirSize;
    unsigned int* allReservoirsExtracted = new unsigned int[segmentSize * _numQueryVectors];
    _myReservoir->extractReservoirs(_numQueryVectors, segmentSize, allReservoirsExtracted, _allQueryHashes);

    _mySketch->add(allReservoirsExtracted, segmentSize);

#ifdef TREE_AGGREGATION
    _mySketch->aggregateSketchesTree();
#endif
#ifdef LINEAR_AGGREGATION
    _mySketch->aggregateSketches();
#endif

    if (_myRank == 0) {
        _mySketch->topK(topK, outputs, threshold);
    }

    delete[] allReservoirsExtracted;
}

void flashControl::localTopK(int topK, unsigned int* outputs, int threshold) {
    int segmentSize = _numTables * _numQueryProbes * _reservoirSize;
    unsigned int* tally = new unsigned int[segmentSize * _numQueryVectors];
    _myReservoir->extractReservoirs(_numQueryVectors, segmentSize, tally, _allQueryHashes);

    _mySketch->add(tally, segmentSize);

    _mySketch->topK(topK, outputs, threshold);

    delete[] tally;
}

void flashControl::printTables() {
    for (int n = 0; n < _worldSize; n++) {
        if (_myRank == n) {
            _myReservoir->tableContents();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void flashControl::showPartitions(){
    printf("[Status Rank %d]:\n\tData Vector Range: [%d, %d)\n\tQuery Vector Range: [%d, %d)\n\tQuery Range: [%d, %d)\n\n", 
            _myRank, 
            _myDataVectorsOffset, _myDataVectorsOffset + _myDataVectorsCt, 
            _queryVectorOffsets[_myRank], _queryVectorOffsets[_myRank] + _myQueryVectorsCt,
            _queryOffsets[_myRank], _queryOffsets[_myRank] + _myQueryVectorsLen);
}

void flashControl::checkQueryHashes() {
    for (int n = 0; n < _worldSize; n++) {
        if (_myRank == n) {
            int hashOffset = _hashOffsets[_myRank];
            printf("Query Hashes Node %d\n", n);
            for (int h = 0; h < _myHashCt; h++) {
                printf("\tHash %d: %d\n", hashOffset + h, _allQueryHashes[hashOffset + h]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (_myRank == 0) {
        printf("\n\nCombined Query Hashes\n");
        for (int h = 0; h < _numQueryVectors * _numTables * _numQueryProbes; h++) {
            printf("\tHash %d: %d\n", h, _allQueryHashes[h]);
        }
    }
}
