#include <cstdio>

#include "util/read_data.hpp"
#include "cnn/for_test/cnn.hpp"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage error!\n");
    printf("====Usage====\n Train: ./cnn train\n Test : ./cnn test\n");
    return 1;
  }
  if (argv[1][1] == 'r')
    CNN::run(TRAIN);
  else
    CNN::run(TEST);
  return 0;
}
  
