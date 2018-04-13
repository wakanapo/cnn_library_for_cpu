#include <cstdio>

#include "util/flags.hpp"
#include "util/read_data.hpp"
#include "cnn/cnn.hpp"
#include "half.hpp"
#include "util/box_quant.hpp"

using half_float::half;
int main(int argc, char* argv[]) {
  Options::ParseCommandLine(argc, argv);
  if (Options::GetType() == HALF) {
    if (Options::IsTrain())
      CNN<half>::run();
    else
      CNN<half>::inference();
  }
  else {
    if (Options::IsTrain())
      CNN<float>::run();
    else
      CNN<float>::inference();
  }
  return 0;
}
  
