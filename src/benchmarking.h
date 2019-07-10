#pragma once


#define WEBSPAM

#ifdef WEBSPAM

#define SPARSE_DATASET

#define NUM_BATCHES				    50
#define BATCH_PRINT                 10

#define NUM_HASHES					4
#define RANGE_POW					15
#define RANGE_ROW_U					15

#define NUM_TABLES					32
#define RESERVOIR_SIZE				64
#define ALLOC_FRACTION				1

#define QUERY_PROBES				1
#define HASHING_PROBES				1

#define DIMENSION					4000
#define FULL_DIMENSION				16609143
// #define NUM_DATA_VECTORS			340000
// #define NUM_QUERY_VECTORS			10000
#define NUM_DATA_VECTORS            10000
#define NUM_QUERY_VECTORS           300
#define MAX_RESERVOIR_RAND			35000
#define TOPK						128
#define AVAILABLE_TOPK				1024

#define AVAILABLE_TOPK				1024
#define TOPK						128

#define CMS_HASHES                  4
#define CMS_BUCKET_SIZE             1024

#define BASEFILE		"../dataset/webspam/trigram.svm"
#define GTRUTHINDICE	"../dataset/webspam/webspam_tri_gtruth_indices.txt"
#define GTRUTHDIST		"../dataset/webspam/webspam_tri_gtruth_distances.txt"

#endif

#ifdef TEST
#define NUM_HASHES 2
#define RANGE_POW 2
#define NUM_TABLES 2
#define RESERVOIR_SIZE 8
#define DIMENSION 5
#define RANGE_ROW_U 2
#define NUM_VECTORS 8
#define NUM_DATA_VECTORS 6
#define NUM_QUERY_VECTORS 2
#define HASHING_PROBES 1
#define QUERY_PROBES 1
#define ALLOC_FRACTION 1
#define TOPK 3

#endif


void controlTest();
void webspamTest();

#if !defined (DENSE_DATASET)
#define SAMFACTOR 24 // DUMMY.
#endif

#if !defined (SPARSE_DATASET)
#define K 10 // DUMMY
#endif
