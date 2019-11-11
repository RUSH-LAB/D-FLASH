#ifndef _BENCHMARKING_H
#define _BENCHMARKING_H

#include <string>

// #define WEBSPAM
#define KDD12

#ifdef WEBSPAM

#define SPARSE_DATASET

#define TREE_AGGREGATION
// #define LINEAR_AGGREGATION

#define CMS_AGGREGATION
// #define BF_AGGREGATION

#define NUM_BATCHES				    50
#define BATCH_PRINT                 10

#define NUM_HASHES					8
#define RANGE_POW					18
#define RANGE_ROW_U					18

// #define NUM_TABLES				    32	
#define NUM_TABLES				    24
#define RESERVOIR_SIZE				256
#define ALLOC_FRACTION				1

#define QUERY_PROBES				1
#define HASHING_PROBES				1

#define DIMENSION					4000
#define FULL_DIMENSION				16609143
#define NUM_DATA_VECTORS			340000
#define NUM_QUERY_VECTORS			10000
// #define NUM_DATA_VECTORS			10000
// #define NUM_QUERY_VECTORS			300
#define MAX_RESERVOIR_RAND			35000
#define TOPK						128
#define AVAILABLE_TOPK				1024

// #define CMS_HASHES                  8
#define CMS_HASHES		            4
// #define CMS_BUCKET_SIZE             2048
#define CMS_BUCKET_SIZE             1024

#define BASEFILE		"../../dataset/webspam/webspam_trigram.svm"
#define GTRUTHINDICE	"../../dataset/webspam/webspam_tri_gtruth_indices.txt"
#define GTRUTHDIST		"../../dataset/webspam/webspam_tri_gtruth_distances.txt"

#endif

#ifdef KDD12

#define SPARSE_DATASET

#define TREE_AGGREGATION
// #define LINEAR_AGGREGATION

#define CMS_AGGREGATION
// #define BF_AGGREGATION

#define NUM_BATCHES				    50
#define BATCH_PRINT                 10

#define NUM_HASHES					4
#define RANGE_POW					18
#define RANGE_ROW_U					18

#define NUM_TABLES				    16
#define RESERVOIR_SIZE				256
#define ALLOC_FRACTION				1

#define QUERY_PROBES				1
#define HASHING_PROBES				1

#define DIMENSION					15
#define FULL_DIMENSION				54686452
// #define NUM_DATA_VECTORS			149000000
// #define NUM_QUERY_VECTORS			10000
#define NUM_DATA_VECTORS			140000000
#define NUM_QUERY_VECTORS			10000
#define MAX_RESERVOIR_RAND			35000
#define TOPK						128
#define AVAILABLE_TOPK				1024

#define CMS_HASHES		            4
#define CMS_BUCKET_SIZE	            2048

#define BASEFILE		"../../dataset/kdd12/kdd12"
#define GTRUTHINDICE	""
#define GTRUTHDIST		""

#endif


#ifdef UNIT_TESTING

#define SPARSE_DATASET

#define NUM_BATCHES				    1
#define BATCH_PRINT                 10

#define NUM_HASHES					4
#define RANGE_POW					8
#define RANGE_ROW_U					8

#define NUM_TABLES				    2	
#define RESERVOIR_SIZE				128
#define ALLOC_FRACTION				1

#define QUERY_PROBES				1
#define HASHING_PROBES				1

#define DIMENSION					4000
#define FULL_DIMENSION				16609143
#define NUM_DATA_VECTORS            64
#define NUM_QUERY_VECTORS           8
#define MAX_RESERVOIR_RAND			35000
#define TOPK						8
#define AVAILABLE_TOPK				1024

#define CMS_HASHES                  2
#define CMS_BUCKET_SIZE             32

#define BASEFILE		"../dataset/webspam/webspam_trigram.svm"
#define GTRUTHINDICE	"../dataset/webspam/webspam_tri_gtruth_indices.txt"
#define GTRUTHDIST		"../dataset/webspam/webspam_tri_gtruth_distances.txt"

#endif

void webspam();
void kdd12();
void showConfig(std::string dataset, int numVectors, int queries, int nodes, int tables, int rangePow, int reservoirSize, int hashes, int cmsHashes, int cmsBucketSize, bool cms);
void unitTesting();

#if !defined (DENSE_DATASET)
#define SAMFACTOR 24 // DUMMY.
#endif

#if !defined (SPARSE_DATASET)
#define K 10 // DUMMY
#endif

#endif
