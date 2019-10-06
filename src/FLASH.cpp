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

int main() {

#ifdef UNIT_TESTING	
	unitTesting();
#endif
#ifdef WEBSPAM
	webspamTest();
#endif
	return 0;
}
