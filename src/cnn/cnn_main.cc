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
    SmallCNNForCifar<half> model;
    if (Options::IsTrain())
      CNN<half>::training(model);
    else
      CNN<half>::inference(model);
  }
  else {
    SmallCNNForCifar<float> model;
    if (Options::IsTrain())
      CNN<float>::training(model);
    else
      CNN<float>::inference(model);
  }
  return 0;
}
  
