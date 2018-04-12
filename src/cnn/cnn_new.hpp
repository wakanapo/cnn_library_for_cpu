#pragma once

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "util/tensor.hpp"
#include "util/function.hpp"
#include "util/flags.hpp"
#include "util/layers.hpp"
#include "util/read_data.hpp"
#include "protos/cnn_params.pb.h"
#include "protos/arithmatic.pb.h"

class CNN {
public:
  CNN(std::vector<Layer> layers) : layers_(layers) {};
  void train();
  void inference();
private:
  std::vector<Layer> layers_;
};

void CNN::train() {
  for (auto layer : layers_) {
    layer.forward();
  }
}
