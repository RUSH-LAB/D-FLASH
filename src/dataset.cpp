#include "dataset.h"

/** For reading sparse matrix dataset in index:value format.
        fileName - name in string
        offset - which datapoint to start reading, normally should be zero
        n - how many data points to read
        indices - array for storing indices
        values - array for storing values
        markers - the start position of each datapoint in indices / values. It
   have length(n + 1), the last position stores start position of the (n+1)th
   data point, which does not exist, but convenient for calculating the length
   of each vector.
*/

void readSparse(std::string fileName, int offset, int n, int *indices,
                float *values, int *markers, unsigned int bufferlen) {
  std::cout << "[readSparse]" << std::endl;

  /* Fill all the markers with the maximum index for the data, to prevent
     indexing outside of the range. */
  for (int i = 0; i <= n; i++) {
    markers[i] = bufferlen - 1;
  }

  std::ifstream file(fileName);
  std::string str;

  unsigned int ct = 0;            // Counting the input vectors.
  unsigned int totalLen = 0;      // Counting all the elements.
  while (std::getline(file, str)) // Get one vector (one vector per line).
  {
    if (ct < offset) { // If reading with an offset, skip < offset vectors.
      ct++;
      continue;
    }
    // Constructs an istringstream object iss with a copy of str as content.
    std::istringstream iss(str);
    // Removes label.
    std::string sub;
    iss >> sub;
    // Mark the start location.
    markers[ct - offset] = std::min(totalLen, bufferlen - 1);
    int pos;
    float val;
    int curLen = 0; // Counting elements of the current vector.
    do {
      std::string sub;
      iss >> sub;
      pos = sub.find_first_of(":");
      if (pos == std::string::npos) {
        continue;
      }
      val = stof(sub.substr(pos + 1, (str.length() - 1 - pos)));
      pos = stoi(sub.substr(0, pos));

      if (totalLen < bufferlen) {
        indices[totalLen] = pos;
        values[totalLen] = val;
      } else {
        std::cout << "[readSparse] Buffer is too small, data is truncated!\n";
        return;
      }
      curLen++;
      totalLen++;
    } while (iss);

    ct++;
    if (ct == (offset + n)) {
      break;
    }
  }
  markers[ct - offset] = totalLen; // Final length marker.
  std::cout << "[readSparse] Read " << totalLen << " numbers, " << ct - offset
            << " vectors. " << std::endl;
}

void writeTopK(std::string filename, int numQueries, int k,
               unsigned int *topK) {
  std::ofstream file;
  file.open(filename);
  for (int q = 0; q < numQueries; q++) {
    for (int i = 0; i < k; i++) {
      file << topK[q * k + i] << " ";
    }
    file << "\n";
  }
  file.close();
}

void readTopK(std::string filename, int numQueries, int k, unsigned int *topK) {
  std::ifstream file(filename);
  std::string str;
  int total = 0;
  while (std::getline(file, str)) {
    std::istringstream iss(str);
    for (int i = 0; i < k; i++) {
      std::string item;
      iss >> item;
      topK[total] = stoi(item);
      total++;
    }
  }
  assert(total == numQueries * k);
  printf("Read top %d vectors for %d Queries\n", k, numQueries);
}

void similarityMetric(int *queries_indice, float *queries_val,
                      int *queries_marker, int *bases_indice, float *bases_val,
                      int *bases_marker, unsigned int *queryOutputs,
                      unsigned int numQueries, unsigned int topk,
                      unsigned int availableTopk, int *nList, int nCnt) {

  float *out_avt = new float[nCnt]();

  std::cout << "[similarityMetric] Averaging output. " << std::endl;
  /* Output average. */
  for (unsigned int i = 0; i < numQueries; i++) {
    int startA, endA;
    startA = queries_marker[i];
    endA = queries_marker[i + 1];
    for (unsigned int j = 0; j < topk; j++) {
      int startB, endB;
      startB = bases_marker[queryOutputs[i * topk + j]];
      endB = bases_marker[queryOutputs[i * topk + j] + 1];
      float dist = cosineDist(queries_indice + startA, queries_val + startA,
                              endA - startA, bases_indice + startB,
                              bases_val + startB, endB - startB);
      for (int n = 0; n < nCnt; n++) {
        if (j < nList[n])
          out_avt[n] += dist;
      }
    }
  }

  /* Print results. */
  printf("\nS@k = s_out(s_true): In top k, average output similarity (average "
         "groundtruth similarity). \n");
  for (unsigned int n = 0; n < nCnt; n++) {
    printf("S@%d = %1.3f \n", nList[n], out_avt[n] / (numQueries * nList[n]));
  }
  for (unsigned int n = 0; n < nCnt; n++)
    printf("%d ", nList[n]);
  printf("\n");
  for (unsigned int n = 0; n < nCnt; n++)
    printf("%1.3f ", out_avt[n] / (numQueries * nList[n]));
  printf("\n");
}
