#include "flashControl.h"

void flashControl::readData(std::string filename, int dataSetSize, int dimension) {
    if (_myRank == 0) {
        _sparseIndices = new int[(unsigned)(dataSetSize * dimension)];
        _sparseVals = new float[(unsigned)(dataSetSize * dimension)];
        _sparseMarkers = new int[(unsigned)(dataSetSize + 1)];
        readSparse(filename, 0, (unsigned)(_numDataVectors + _numQueryVectors), _sparseIndices, _sparseVals, _sparseMarkers, (unsigned)((_numDataVectors + _numQueryVectors) * dimension));
        //dummyReadSparse(filename, 0, (unsigned)(_numDataVectors + _numQueryVectors), _sparseIndices, _sparseVals, _sparseMarkers, (unsigned)((_numDataVectors + _numQueryVectors) * dimension));

        for (int n = 0; n < _worldSize; n++) {
            _dataOffsets[n] = _sparseMarkers[_dataVectorOffsets[n]];
            _dataCts[n] = _sparseMarkers[_dataVectorOffsets[n] + _dataVectorCts[n]] - _dataOffsets[n];
            _queryOffsets[n] = _sparseMarkers[_queryVectorOffsets[n]];
            _queryCts[n] = _sparseMarkers[_queryVectorOffsets[n] + _queryVectorCts[n]] - _queryOffsets[n];
        }
    }

#ifdef DEBUG
    if (_myRank == 0) {
        printf("Data and Query Counts and Offsets...\n");
        for(int i = 0; i < _worldSize; i++) {
            printf("[Rank %d]: Data Offset: %d, Data Ct: %d, Query Offset: %d, Query Ct: %d\n", i, _dataOffsets[i], _dataCts[i], _queryOffsets[i], _queryCts[i]);
        }
    }
#endif

    MPI_Bcast(_dataOffsets, _worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(_dataCts, _worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(_queryOffsets, _worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(_queryCts, _worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    
    _myDataVectorsLen = _dataCts[_myRank];
    _myDataOffset = _dataOffsets[_myRank];
    _myQueryVectorsLen = _queryCts[_myRank];
}

void flashControl::allocateData() {
    _myDataIndices = new int[_myDataVectorsLen];
    _myDataVals = new float[_myDataVectorsLen];
    _myDataMarkers = new int[_myDataVectorsCt + 1];

    int* tempDataMarkerCts = new int[_worldSize];
    for (int n = 0; n < _worldSize; n++) {
        tempDataMarkerCts[n] = _dataVectorCts[n] + 1; // To account for extra element at the end of each marker array
    }

    MPI_Scatterv(_sparseIndices, _dataCts, _dataOffsets, MPI_INT, _myDataIndices, _myDataVectorsLen, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(_sparseVals, _dataCts, _dataOffsets, MPI_FLOAT, _myDataVals, _myDataVectorsLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(_sparseMarkers, tempDataMarkerCts, _dataVectorOffsets, MPI_INT, _myDataMarkers, _myDataVectorsCt + 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < _myDataVectorsCt + 1; i++) {
        _myDataMarkers[i] -= _myDataOffset;
    }
    delete[] tempDataMarkerCts;
}

void flashControl::allocateQuery() {
    _myQueryIndices = new int[_myQueryVectorsLen];
    _myQueryVals = new float[_myQueryVectorsLen];
    _myQueryMarkers = new int[_myQueryVectorsCt + 1];

    int* tempQueryMarkerCts = new int[_worldSize];
    for (int n = 0; n < _worldSize; n++) {
        tempQueryMarkerCts[n] = _queryVectorCts[n] + 1; // To account for extra element at the end of each marker array
    }

    MPI_Scatterv(_sparseIndices, _queryCts, _queryOffsets, MPI_INT, _myQueryIndices, _myQueryVectorsLen, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(_sparseVals, _queryCts, _queryOffsets, MPI_FLOAT, _myQueryVals, _myQueryVectorsLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(_sparseMarkers, tempQueryMarkerCts, _queryVectorOffsets, MPI_INT, _myQueryMarkers, _myQueryVectorsCt + 1, MPI_INT, 0, MPI_COMM_WORLD);

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
    for (int partition = 0; partition < _worldSize; partition++) {
        len = _queryVectorCts[partition] * _numQueryProbes;
        for (int tb = 0; tb < _numTables; tb++) {
            unsigned int* old = queryHashBuffer + _hashOffsets[partition] + tb * len;
            unsigned int* fin = _allQueryHashes + tb * _numQueryVectors * _numQueryProbes + (_hashOffsets[partition] / _numTables);
            for (int l = 0; l < len; l++) {
                fin[l] = old[l];
            }
        }
    }

    delete[] queryHashBuffer;
    delete[] myPartitionHashes;
}

void flashControl::extractReservoirs(int topK, unsigned int* outputs) {
    _myReservoir->extractTopK(_numQueryVectors, _allQueryHashes, topK, outputs);
}

void flashControl::extractReservoirsCMS(int topK, unsigned int* outputs, int threshold) {
    int segmentSize = _numTables * _numQueryProbes * _reservoirSize;
    unsigned int* tally = new unsigned int[segmentSize * _numQueryVectors];
    _myReservoir->extractReservoirs(_numQueryVectors, segmentSize, tally, _allQueryHashes);

    _mySketch->add(tally, segmentSize);

    //_mySketch->printBuckets();

    _mySketch->aggregateSketches();

    if (_myRank == 0) {
        _mySketch->topK(topK, outputs, threshold);
    }

    delete[] tally;
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
    printf("[Status Rank %d]:\n\tData Vector Range: [%d, %d)\n\tData Range: [%d, %d)\n\tQuery Vector Range: [%d, %d)\n\tQuery Range: [%d, %d)\n\n", 
            _myRank, 
            _myDataVectorsOffset, _myDataVectorsOffset + _myDataVectorsCt, 
            _dataOffsets[_myRank], _dataOffsets[_myRank] + _myDataVectorsLen,
            _queryVectorOffsets[_myRank], _queryVectorOffsets[_myRank] + _myQueryVectorsCt,
            _queryOffsets[_myRank], _queryOffsets[_myRank] + _myQueryVectorsLen);
}

void flashControl::checkDataTransfer() {
    std::cout << "Markers Check" << std::endl; 
    if (_myRank == 0) {
        std::cout << "\nInitial Read" << std::endl;
        for (int i = 0; i < _numDataVectors + _numQueryVectors + 1; i++) {
            printf("\t%d. %d\n", i, _sparseMarkers[i]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int n = 0; n < _worldSize; n++) {
        if (_myRank == n) {
            printf("\nQuery Markers Node %d\n", n);
            for (int i = 0; i < _myQueryVectorsCt + 1; i++) {
                printf("\t%d. %d\n", i, _myQueryMarkers[i] + _queryOffsets[_myRank]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    for (int n = 0; n < _worldSize; n++) {
        if (_myRank == n) {
            printf("\nData Markers Node %d\n", n);
            for (int i = 0; i < _myDataVectorsCt + 1; i++) {
                printf("\t%d. %d\n", i, _myDataMarkers[i]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

    }
    std::cout << "Indices Check" << std::endl;
    if (_myRank == 0) {
        std::cout << "\nInitial Read" << std::endl;
        for (int i = 0; i < _numDataVectors + _numQueryVectors; i++) {
            printf("\t%d. Start: %d  Middle: %d  End: %d\n", i, _sparseIndices[_sparseMarkers[i]], _sparseIndices[_sparseMarkers[i] + 12], _sparseIndices[_sparseMarkers[i+1]]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int n = 0; n < _worldSize; n++) {
        if (_myRank == n) {
            printf("\nQuery Indices Node %d\n", n);
            for (int i = 0; i < _myQueryVectorsCt; i++) {
                printf("\t%d. Start: %d  Middle: %d  End: %d\n", i, _myQueryIndices[_myQueryMarkers[i]], _myQueryIndices[_myQueryMarkers[i] + 12], _myQueryIndices[_myQueryMarkers[i+1]]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    for (int n = 0; n < _worldSize; n++) {
        if (_myRank == n) {
            printf("\nData Indices Node %d\n", n);
            for (int i = 0; i < _myDataVectorsCt; i++) {
                printf("\t%d. Start: %d  Middle: %d  End: %d\n", i, _myDataIndices[_myDataMarkers[i]], _myDataIndices[_myDataMarkers[i] + 12], _myDataIndices[_myDataMarkers[i+1]]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
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
