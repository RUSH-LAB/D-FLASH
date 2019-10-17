#include "flashControl.h"

flashControl::flashControl(LSHReservoirSampler* reservoir, CMS* cms, int myRank, int worldSize, int numDataVectors, int numQueryVectors, int dimension, int numTables, int numQueryProbes, int reservoirSize) {

    // Core Params
    _myReservoir = reservoir;
    _mySketch = cms;
    _myRank = myRank;
    _worldSize = worldSize;
    _numDataVectors = numDataVectors;
    _numQueryVectors = numQueryVectors;
    _dimension = dimension;
    _numTables = numTables;
    _numQueryProbes = numQueryProbes;
    _reservoirSize = reservoirSize;

    _dataVectorOffsets = new int[_worldSize];
    _dataVectorCts = new int[_worldSize]();
    // _dataOffsets = new int[_worldSize];
    // _dataCts = new int[_worldSize];

    _queryVectorOffsets = new int[_worldSize];
    _queryVectorCts = new int[_worldSize]();
    _queryOffsets = new int[_worldSize];
    _queryCts = new int[_worldSize];

    _hashCts = new int[_worldSize];
    _hashOffsets = new int[_worldSize];

    int dataPartitionSize = std::floor((float) _numDataVectors / (float) _worldSize);
    int dataPartitionRemainder = _numDataVectors % _worldSize;
    int queryPartitionSize = std::floor((float) _numQueryVectors / (float) _worldSize);
    int queryPartitionRemainder = _numQueryVectors % _worldSize;

    for (int i = 0; i < _worldSize; i++) {
        _dataVectorCts[i] = dataPartitionSize;
        if (i < dataPartitionRemainder) {
            _dataVectorCts[i] ++;
        }
    }

    for (int j = 0; j < _worldSize; j++) {
        _queryVectorCts[j] = queryPartitionSize;
        if (j < queryPartitionRemainder) {
            _queryVectorCts[j] ++;
        }
        _hashCts[j] = _queryVectorCts[j] * _numQueryProbes * _numTables;
    }
    _dataVectorOffsets[0] = _numQueryVectors;
    _queryVectorOffsets[0] = 0;
    _hashOffsets[0] = 0;
    for (int n = 1; n < _worldSize; n++) {
        _dataVectorOffsets[n] = std::min(_dataVectorOffsets[n-1] + _dataVectorCts[n-1], _numDataVectors + _numQueryVectors - 1); // Overflow prevention
        _queryVectorOffsets[n] = std::min(_queryVectorOffsets[n-1] + _queryVectorCts[n-1], _numQueryVectors - 1); // Overflow prevention
        _hashOffsets[n] = _hashOffsets[n-1] + _hashCts[n-1];
    }

    _allQueryHashes = new unsigned int[_numQueryVectors * _numQueryProbes * _numTables];

    _myDataVectorsCt = _dataVectorCts[_myRank];
    _myDataVectorsOffset = _dataVectorOffsets[_myRank];
    _myQueryVectorsCt = _queryVectorCts[_myRank];
    _myHashCt = _hashCts[_myRank];

    std::cout << "FLASH Controller Initialized in Node " << _myRank << std::endl;
}

flashControl::~flashControl() {
    delete[] _dataVectorOffsets;
    delete[] _dataVectorCts;

    delete[] _queryVectorOffsets;
    delete[] _queryVectorCts;
    delete[] _queryOffsets;
    delete[] _queryCts;

    delete[] _myDataIndices;
    delete[] _myDataVals;
    delete[] _myDataMarkers;

    delete[] _myQueryIndices;
    delete[] _myQueryVals;
    delete[] _myQueryMarkers;

    delete[] _hashCts;
    delete[] _hashOffsets;

    if (_myRank == 0) {
        delete[] _queryIndices;
        delete[] _queryVals;
        delete[] _queryMarkers;
    }

    delete[] _allQueryHashes;
}
