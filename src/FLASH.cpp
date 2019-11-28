#include "LSHReservoirSampler.h"
#include "flashControl.h"
#include "CMS.h"
#include "LSH.h"
#include "dataset.h"
#include "mathUtils.h"
#include "indexing.h"
#include "omp.h"
#include "benchmarking.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>

int main() {

#ifdef UNIT_TESTING	
	unitTesting();
#endif
#ifdef WEBSPAM
	webspam();	
#endif
#ifdef KDD12
	kdd12FileOutput();	
#endif
	return 0;
}
