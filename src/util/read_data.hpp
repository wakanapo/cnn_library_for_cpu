#pragma once

#include <iostream>
#include <cstdlib>
#include <tuple>
#include <memory>
#include <vector>

#include "util/flags.hpp"
#include "util/tensor.hpp"

namespace {
  const char* kMnistTestImageFilePath = "data/mnist/t10k-images-idx3-ubyte";
  const char* kMnistTrainImageFilePath = "data/mnist/train-images.idx3-ubyte";
  const char* kMnistTestLabelFilePath = "data/mnist/t10k-labels-idx1-ubyte";
  const char* kMnistTrainLabelFilePath = "data/mnist/train-labels.idx1-ubyte";
  const int kExpectImageMagicNumber = 2051;
  const int kExpectLabelMagicNumber = 2049;

  const char* kCifar100TestDataFilePath = "data/cifar-100-binary/test.bin";
  const char* kCifar100TrainDataFilePath = "data/cifar-100-binary/train.bin";
  const char* kCifar10TestDataFilePath = "data/cifar-10-binary/test.bin";
  const char* kCifar10TrainDataFilePath = "data/cifar-10-binary/train.bin";
}

enum Status {
  TRAIN,
  TEST
};

template<typename ImageType, typename LabelType>
struct Dataset {
  std::vector<ImageType> images;
  std::vector<LabelType> labels;
};

int ConvertEndian(int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int) c2 << 16) + ((int)c3 << 8) + c4;
}

template<typename LabelType>
LabelType OneHot(unsigned long t) {
  LabelType onehot;
  for (int i = 0; i < onehot.size(); ++i) {
    onehot[i] = (i == (int)t) ? 1 : 0;
  }
  return onehot;
}

template<typename ImageType>
std::vector<ImageType> ReadMnistImages(Status st) {
  FILE *fp = fopen((st == TRAIN) ? kMnistTrainImageFilePath :
                   kMnistTestImageFilePath, "rb");
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

  std::vector<ImageType> images;
  for (int n = 0; n < number_of_images; ++n) {
    ImageType image;
    for (int i = 0; i < number_of_cols; ++i) {
      for (int j = 0; j < number_of_rows; ++j) {
        unsigned char temp = 0;
        err = fread(&temp, sizeof(temp), 1, fp);
        if (err < 1)
          std::cerr << "File read error!" << std::endl;
        image[i * number_of_rows + j] = temp / 255.0;
      }
    }
    images.push_back(image);
  }
  fclose(fp);
  return images;
}

template<typename LabelType>
std::vector<LabelType> ReadMnistLabels(Status st) {
  FILE *fp = fopen((st == TRAIN) ? kMnistTrainLabelFilePath :
                   kMnistTestLabelFilePath, "rb");
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

  std::vector<LabelType> labels;
  for (int i = 0; i < number_of_labels; ++i) {
    LabelType label;
    unsigned char temp = 0;
    err = fread(&temp, sizeof(temp), 1, fp);
    if (err < 1)
      std::cerr << "File read error!" << std::endl;
    label = OneHot<LabelType>((unsigned long)temp);
    labels.push_back(label);
  }
  fclose(fp);
  return labels;
}

template<typename ImageType, typename LabelType>
Dataset<ImageType, LabelType> ReadMNISTData(Status st) {
  Dataset<ImageType, LabelType> data =
    {ReadMnistImages<ImageType>(st), ReadMnistLabels<LabelType>(st)};
  std::cerr << "Success read MNIST " <<
    (st==TRAIN ? "Train" : "Test") << " data." << std::endl;
  return data;
}

enum CifarClass {
  COARSE,
  FINE
};

template<typename ImageType, typename LabelType>
Dataset<ImageType, LabelType> ReadCifar100Data(Status st, const CifarClass c_class) {
  FILE *fp =  fopen((st == TRAIN) ? kCifar100TrainDataFilePath :
                    kCifar100TestDataFilePath, "rb");
  if (fp == NULL) {
    std::cerr << "File open error!!" << std::endl;
    exit(EXIT_FAILURE);
  }

  int number_of_images = (st == TRAIN) ? 50000 : 10000;
  int number_of_rows = 32;
  int number_of_cols = 32;
  int number_of_channels = 3;

  int image_2d = number_of_rows * number_of_cols;

  std::vector<ImageType> images;
  std::vector<LabelType> labels;

  for (int n = 0; n < number_of_images; ++n) {
    ImageType image;
    LabelType label;
    unsigned char temp = 0;
    size_t err = fread(&temp, sizeof(temp), 1, fp);
    if (err < 1) {
      std::cerr << "File read error!" << std::endl;
      exit(1);
    }
    if (c_class == COARSE)
      label = OneHot<LabelType>((unsigned long)temp);
    err = fread(&temp, sizeof(temp), 1, fp);
    if (err < 1) {
      std::cerr << "File read error!" << std::endl;
      exit(1);
    }
    if (c_class == FINE)
      label = OneHot<LabelType>((unsigned long)temp);

    for (int i = 0; i < number_of_channels; ++i) {
      for (int j = 0; j < number_of_cols; ++j) {
        for (int k = 0; k < number_of_rows; ++k) {
          err = fread(&temp, sizeof(temp), 1, fp);
          if (err < 1) {
            std::cerr << "File read error!" << std::endl;
            exit(1);
          }
          image[i * image_2d + j * number_of_rows + k]
            = temp / 255.0;
        }
      }
    }
    labels.push_back(label);
    images.push_back(image);
  }
  fclose(fp);
  std::cerr << "Success read Cifar100 " <<
    (st==TRAIN ? "Train" : "Test") << " data." << std::endl;
  return {images, labels};
}


template<typename ImageType, typename LabelType>
Dataset<ImageType, LabelType> ReadCifar10Data(Status st) {
  FILE *fp =  fopen((st == TRAIN) ? kCifar10TrainDataFilePath :
                    kCifar10TestDataFilePath, "rb");
  if (fp == NULL) {
    std::cerr << "File open error!!" << std::endl;
    exit(EXIT_FAILURE);
  }

  int number_of_images = (st == TRAIN) ? 50000 : 10000;
  int number_of_rows = 32;
  int number_of_cols = 32;
  int number_of_channels = 3;

  int image_2d = number_of_rows * number_of_cols;

  std::vector<ImageType> images;
  std::vector<LabelType> labels;

  for (int n = 0; n < number_of_images; ++n) {
    ImageType image;
    LabelType label;
    unsigned char temp = 0;
    size_t err = fread(&temp, sizeof(temp), 1, fp);
    if (err < 1) {
      std::cerr << "File read error!" << std::endl;
      exit(1);
    }
    label = OneHot<LabelType>((unsigned long)temp);

    for (int i = 0; i < number_of_channels; ++i) {
      for (int j = 0; j < number_of_cols; ++j) {
        for (int k = 0; k < number_of_rows; ++k) {
          err = fread(&temp, sizeof(temp), 1, fp);
          if (err < 1) {
            std::cerr << "File read error!" << std::endl;
            exit(1);
          }
          image[i * image_2d + j * number_of_rows + k]
            = temp / 255.0;
        }
      }
    }
    labels.push_back(label);
    images.push_back(image);
  }
  fclose(fp);
  std::cerr << "Success read Cifar10 " <<
    (st==TRAIN ? "Train" : "Test") << " data." << std::endl;
  return {images, labels};
}
