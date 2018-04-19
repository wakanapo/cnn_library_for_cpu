#pragma once

#include <iostream>
#include <cstdlib>
#include <memory>

#include "util/flags.hpp"

namespace {
  const char* kMnistTestImageFilePath = "data/mnist/t10k-images-idx3-ubyte";
  const char* kMnistTrainImageFilePath = "data/mnist/train-images.idx3-ubyte";
  const char* kMnistTestLabelFilePath = "data/mnist/t10k-labels-idx1-ubyte";
  const char* kMnistTrainLabelFilePath = "data/mnist/train-labels.idx1-ubyte";
  const int kExpectImageMagicNumber = 2051;
  const int kExpectLabelMagicNumber = 2049;

  const char* kCifar100TestDataFilePath = "data/cihar-100-binary/test.bin";
  const char* kCifar100TrainDataFilePath = "data/cihar-100-binary/test.bin";
}

enum status {
  TRAIN,
  TEST
};

template<typename T>
class Data {
public:
  Data(int row, int col, T* ptr) : row_(row), col_(col), ptr_(ptr) {};
  int row_;
  int col_;
  T* ptr_;
};

int ConvertEndian(int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int) c2 << 16) + ((int)c3 << 8) + c4;
}

template<typename T>
Data<T> ReadMnistImages(status st) {
  FILE *fp = (st == TRAIN) ? fopen(kMnistTrainImageFilePath, "rb") :
    fopen(kMnistTestImageFilePath, "rb");
  if (fp == NULL) {
    std::cerr << "File open error!!" << std::endl;
    exit(EXIT_FAILURE);
  }

  int magic_number = 0;
  int number_of_images = 0;
  int number_of_rows = 0;
  int number_of_cols = 0;

  size_t err = fread(&magic_number, sizeof(int), 1, fp);
  magic_number = ConvertEndian(magic_number);
  if (magic_number != kExpectImageMagicNumber) {
    std::cerr << "Invalid MNIST image file!" << std::endl;
    exit(EXIT_FAILURE);
  }

  err = fread(&number_of_images, sizeof(int), 1, fp);
  number_of_images = ConvertEndian(number_of_images);
  err = fread(&number_of_rows, sizeof(int), 1, fp);
  number_of_rows = ConvertEndian(number_of_rows);
  err = fread(&number_of_cols, sizeof(int), 1, fp);
  number_of_cols = ConvertEndian(number_of_cols);
  int image_size = number_of_rows * number_of_cols;

  T* datasets =
    (T*)malloc(sizeof(T) * number_of_images * image_size);
  for (int n = 0; n < number_of_images; ++n) {
    for (int i = 0; i < number_of_cols; ++i) {
      for (int j = 0; j < number_of_rows; ++j) {
        unsigned char temp = 0;
        err = fread(&temp, sizeof(temp), 1, fp);
        if (err < 1)
          std::cerr << "File read error!" << std::endl;
        datasets[n * image_size + i * number_of_rows + j] = (T)temp / 255.0;
      }
    }
  }
  fclose(fp);
  Data<T> images(image_size, number_of_images, datasets);
  return images;
}

Data<unsigned long> ReadMnistLabels(status st) {
  FILE *fp = (st == TRAIN) ? fopen(kMnistTrainLabelFilePath, "rb") :
    fopen(kMnistTestLabelFilePath, "rb");
  if (fp == NULL) {
    std::cerr << "File open error!!" << std::endl;
    exit(EXIT_FAILURE);
  }

  int magic_number = 0;
  int number_of_labels = 0;
  size_t err = fread(&magic_number, sizeof(int), 1, fp);
  magic_number = ConvertEndian(magic_number);
  if (magic_number != kExpectLabelMagicNumber) {
    std::cerr << "Invalid MNIST label file!" << std::endl;
    exit(EXIT_FAILURE);
  }
  err = fread(&number_of_labels, sizeof(int), 1, fp);
  number_of_labels = ConvertEndian(number_of_labels);

  unsigned long* datasets =
    (unsigned long*)malloc(sizeof(unsigned long) * number_of_labels);
  for (int i = 0; i < number_of_labels; ++i) {
    unsigned char temp = 0;
    err = fread(&temp, sizeof(temp), 1, fp);
    if (err < 1)
      std::cerr << "File read error!" << std::endl;
    datasets[i] = (unsigned long)temp;
  }
  fclose(fp);
  Data<unsigned long> labels(1, number_of_labels, datasets);
  return labels;
}

template<typename T>
T* mnistOneHot(unsigned long t) {
  T* onehot = (T*)malloc(10*sizeof(T));
  for (int i = 0; i < 10; ++i) {
    onehot[i] = (i == (int)t) ? 1 : 0;
  }
  return onehot;
}

template<typename T>
Data<T> ReadCifar100Data(status st) {
  FILE *fp = (st == TRAIN) ? fopen(kCifar100TrainDataFilePath, "rb") :
    fopen(kCifar100TestDataFilePath, "rb");
  if (fp == NULL) {
    std::cerr << "File open error!!" << std::endl;
    exit(EXIT_FAILURE);
  }

  int number_of_images = (st == TRAIN) ? 50000 : 10000;
  int number_of_rows = 32;
  int number_of_cols = 32;
  int number_of_channels = 3;

  int image_2d = number_of_rows * number_of_cols;
  int image_3d = number_of_rows * number_of_cols * number_of_channels;

  T* datasets =
    (T*)malloc(sizeof(T) * number_of_images * image_3d);
  for (int n = 0; n < number_of_images; ++n) {
    for (int i = 0; i < number_of_channels; ++i) {
      for (int j = 0; j < number_of_cols; ++j) {
        for (int k = 0; k < number_of_rows; ++k) {
          unsigned char temp = 0;
          size_t err = fread(&temp, sizeof(temp), 1, fp);
          if (err < 1)
            std::cerr << "File read error!" << std::endl;
          datasets[n * image_3d + i * image_2d + j * number_of_rows + k]
            = (T)temp / 255.0;
        }
      }
    }
  }
  fclose(fp);
  Data<T> images(image_3d, number_of_images, datasets);
  return images;
}
