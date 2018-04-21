#include <cstdio>

#include "util/flags.hpp"
#include "util/read_data.hpp"
#include "cnn/cnn.hpp"
#include "cnn/vgg.hpp"
#include "cnn/small_cnn_for_cifar.hpp"
#include "cnn/simple_conv_net.hpp"
#include "half.hpp"
#include "util/box_quant.hpp"

using half_float::half;
template<typename T>
using Model = VGG16<T>;

int main(int argc, char* argv[]) {
  Options::ParseCommandLine(argc, argv);
  if (Options::GetType() == HALF) {
    if (Options::IsTrain())
      CNN<Model<half>>::training();
    else
      CNN<Model<half>>::inference();
  }
  else {
    if (Options::IsTrain())
      CNN<Model<float>>::training();
    else
      CNN<Model<float>>::inference();
  }
  return 0;
}
  
