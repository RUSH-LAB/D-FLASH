#include "CMS.h"
#include "LSH.h"
#include "LSHReservoirSampler.h"
#include "benchmarking.h"
#include "dataset.h"
#include "flashControl.h"
#include "indexing.h"
#include "mathUtils.h"
#include "omp.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {

  // #ifdef UNIT_TESTING
  // 	unitTesting();
  // #endif
  // #ifdef WEBSPAM
  // 	webspam();
  // #endif
  // #ifdef KDD12
  // 	kdd12FileOutput();
  // #endif

  evaluateResults("../results/kdd12/Bruteforce-Nodes-3");
  return 0;
}
