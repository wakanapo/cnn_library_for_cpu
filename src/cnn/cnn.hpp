#pragma once

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <fcntl.h>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "util/flags.hpp"
#include "util/layers.hpp"
#include "util/progress_bar.hpp"
#include "util/read_data.hpp"
#include "protos/cnn_params.pb.h"
#include "protos/arithmatic.pb.h"

#define CPU_NUM 4

template <typename ModelType>
class CNN {
public:
  using Type = typename ModelType::Type;
  using InputType = typename ModelType::InputType;
  using OutputType = typename ModelType::OutputType;
  static void training();
  static void inference();
};

template <typename ModelType>
void CNN<ModelType>::training() {
  ModelType model;
  Dataset<InputType, OutputType> train = model.readData(TRAIN);
  Dataset<InputType, OutputType> test = model.readData(TEST);

  Type eps = (Type)0.01;
  int epoch = 10;
  int image_num = 5000;

  for (int k = 0; k < epoch; ++k) {
    std::cout << "--------epoch " << k << "--------" << std::endl;
    for (int i = image_num*k; progressBar(i-image_num*k, image_num); ++i) {
      if (Options::IsSaveArithmetic()) {
        std::stringstream sFile;
        sFile << Options::GetArithmaticOutput();
        model.train(train.images[i], train.labels[i], eps);
        using namespace google::protobuf::io;
        std::ofstream output(sFile.str(), std::ofstream::out | std::ofstream::trunc
                             | std::ofstream::binary);
        OstreamOutputStream outputFileStream(&output);
        GzipOutputStream::Options options;
        options.format = GzipOutputStream::GZIP;
        options.compression_level = 9;
        GzipOutputStream gzip_stream(&outputFileStream, options);
        if (!(p.SerializeToZeroCopyStream(&gzip_stream))) {
          std::cerr << "Failed to write values." << std::endl;
        }
        p.Clear();
      }
      else {
        model.train(train.images[i], train.labels[i], eps);
      }
    }
    std::cout << "Finish Train." << std::endl;
    
    std::vector<std::future<int>> futures;
    auto start = std::chrono::system_clock::now();
    for (int cpu = 0; cpu < CPU_NUM; ++ cpu) {
      futures.push_back(std::async(std::launch::async, [&] {
            int cnt = 0;
#pragma clang loop unroll(full)
            for (int i = 0; i < 4096 / CPU_NUM; ++i) {
              unsigned long y = model.predict(test.images[i]);
              if (Options::IsSaveArithmetic())
                p.Clear();
              if (OneHot<OutputType>(y) == test.labels[i])
                ++cnt;
            }
            return cnt;
          }));
    }
    int sum = 0;
    for (auto &f : futures) {
      sum += f.get();
    }
    auto end = std::chrono::system_clock::now();
    auto diff = end - start;
    std::cout << "Inference time = "
              << std::chrono::duration_cast<std::chrono::seconds>(diff).count()
              << " sec."
              << std::endl;
    std::cout << "Accuracy: " << (float)sum / (float)4096 << std::endl;
  }
  if (Options::IsSaveParams())
    model.save();
}

template <typename ModelType>
void CNN<ModelType>::inference() {
  ModelType model;
  Dataset<InputType, OutputType> test = model.readData(TEST);

  model.load();
  int cnt = 0;
  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < 3000; ++i) {
    unsigned long y = model.predict(test.images[i]);
    if (OneHot<OutputType>(y) == test.labels[i])
      ++cnt;
  }
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;
  std::cout << "Inference time = "
            << std::chrono::duration_cast<std::chrono::seconds>(diff).count()
            << " sec."
            << std::endl;
  std::cout << "Accuracy: " << (float)cnt / (float)3000 << std::endl;
}


