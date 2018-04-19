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
      CNN<SmallCNNForCifar<half>>::training();
    else
      CNN<SmallCNNForCifar<half>>::inference();
  }
  else {
    if (Options::IsTrain())
      CNN<SmallCNNForCifar<float>>::training();
    else
      CNN<SmallCNNForCifar<float>>::inference();
  }
  return 0;
}
  
