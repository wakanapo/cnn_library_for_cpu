#pragma once

#include <iostream>
#include <cstdlib>

namespace {
  const char* kMnistTestImageFilePath = "data/t10k-images-idx3-ubyte";
  const char* kMnistTrainImageFilePath = "data/train-images.idx3-ubyte";
  const char* kMnistTestLabelFilePath = "data/t10k-labels-idx1-ubyte";
  const char* kMnistTrainLabelFilePath = "data/train-labels.idx1-ubyte";
  const int kExpectImageMagicNumber = 2051;
  const int kExpectLabelMagicNumber = 2049;
}

enum Status {
  TRAIN,
  TEST
};

class Data {
public:
  Data(int col, int row, void* ptr) : col_(col), row_(row), ptr_(ptr) {};
  const int col_;
  const int row_;
  void* ptr_;
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
Data ReadMnistImages(Status st) {
  FILE *fp = (st == TEST) ? fopen(kMnistTestImageFilePath, "rb") :
    fopen(kMnistTrainImageFilePath, "rb");
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
    for (int i = 0; i < number_of_rows; ++i) {
      for (int j = 0; j < number_of_cols; ++j) {
        unsigned char temp = 0;
        err = fread(&temp, sizeof(temp), 1, fp);
        if (err < 1)
          std::cerr << "File read error!" << std::endl;
        datasets[n * image_size + i * number_of_cols + j] = (T)temp / 255.0;
      }
    }
  }
  fclose(fp);
  Data images(number_of_images, image_size, datasets);
  return images;
}

Data ReadMnistLabels(Status st) {
  FILE *fp = (st == TEST) ? fopen(kMnistTestLabelFilePath, "rb") :
    fopen(kMnistTrainLabelFilePath, "rb");
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
  Data labels(1, number_of_labels, datasets);
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
