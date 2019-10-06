#ifndef _BENCHMARKING_H
#define _BENCHMARKING_H


// #define WEBSPAM
#define UNIT_TESTING

#ifdef WEBSPAM

#define SPARSE_DATASET

#define NUM_BATCHES				    50
#define BATCH_PRINT                 10

#define NUM_HASHES					4
#define RANGE_POW					15
#define RANGE_ROW_U					15

#define NUM_TABLES				    16	
#define RESERVOIR_SIZE				64
#define ALLOC_FRACTION				1

#define QUERY_PROBES				1
#define HASHING_PROBES				1

#define DIMENSION					4000
#define FULL_DIMENSION				16609143
//#define NUM_DATA_VECTORS			340000
//#define NUM_QUERY_VECTORS			10000
#define NUM_DATA_VECTORS            10000
#define NUM_QUERY_VECTORS           300
#define MAX_RESERVOIR_RAND			35000
#define TOPK						128
#define AVAILABLE_TOPK				1024

#define CMS_HASHES                  4
#define CMS_BUCKET_SIZE             256

#define BASEFILE		"../../dataset/webspam/webspam_trigram.svm"
#define GTRUTHINDICE	"../../dataset/webspam/webspam_tri_gtruth_indices.txt"
#define GTRUTHDIST		"../../dataset/webspam/webspam_tri_gtruth_distances.txt"

#endif


#ifdef UNIT_TESTING

#define SPARSE_DATASET

#define NUM_BATCHES				    1
#define BATCH_PRINT                 10

#define NUM_HASHES					4
#define RANGE_POW					8
#define RANGE_ROW_U					8

#define NUM_TABLES				    2	
#define RESERVOIR_SIZE				16
#define ALLOC_FRACTION				1

#define QUERY_PROBES				1
#define HASHING_PROBES				1

#define DIMENSION					4000
#define FULL_DIMENSION				16609143
#define NUM_DATA_VECTORS            128
#define NUM_QUERY_VECTORS           8
#define MAX_RESERVOIR_RAND			35000
#define TOPK						8
#define AVAILABLE_TOPK				1024

#define CMS_HASHES                  2
#define CMS_BUCKET_SIZE             32

#define BASEFILE		"../../dataset/webspam/webspam_trigram.svm"
#define GTRUTHINDICE	"../../dataset/webspam/webspam_tri_gtruth_indices.txt"
#define GTRUTHDIST		"../../dataset/webspam/webspam_tri_gtruth_distances.txt"

#endif

#ifdef TEST
#define NUM_HASHES 2
#define RANGE_POW 2
#define NUM_TABLES 2
#define RESERVOIR_SIZE 8
#define DIMENSION 5
#define RANGE_ROW_U 2
#define NUM_VECTORS 8
#define NUM_DATA_VECTORS 4
#define NUM_QUERY_VECTORS 4
#define HASHING_PROBES 1
#define QUERY_PROBES 1
#define ALLOC_FRACTION 1
#define TOPK 3
#define AVAILABLE_TOPK 8
#define CMS_HASHES 2
#define CMS_BUCKET_SIZE 8

//Dummy Variables
#define BASEFILE ""
#define GTRUTHINDICE ""
#define GTRUTHDIST ""
#define NUM_BATCHES 1
#define BATCH_PRINT 1

#endif


void controlTest();
void webspamTest();
void localSimilarityTest();
void unitTesting();

#if !defined (DENSE_DATASET)
#define SAMFACTOR 24 // DUMMY.
#endif

#if !defined (SPARSE_DATASET)
#define K 10 // DUMMY
#endif

#endif
