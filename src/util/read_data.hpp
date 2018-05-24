#pragma once

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <tuple>
#include <memory>
#include <string>
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

  const std::string kCifar100TestDataFilePath = "data/cifar-100-binary/test.bin";
  const std::string kCifar100TrainDataFilePath = "data/cifar-100-binary/train.bin";
  const std::string kCifar10TestDataFilePath = "data/cifar-10-binary/test.bin";
  const std::string kCifar10TrainDataFilePath = "data/cifar-10-binary/train.bin";
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
  std::fstream ifs((st == TRAIN) ? kCifar100TrainDataFilePath :
                   kCifar100TestDataFilePath, std::ios::binary);
  if (!ifs) {
    std::cerr << "File open error!!" << std::endl;
    exit(EXIT_FAILURE);
  }

  int number_of_images = (st == TRAIN) ? 50000 : 10000;
  int image_size = 32 * 32 * 3;
<<<<<<< f49c590708b62703a74ba695516b4a2f1e20dfbd

  std::vector<ImageType> images(number_of_images);
  std::vector<LabelType> labels(number_of_images);
=======
>>>>>>> Change to use fstream.

  for (int i = 0; i < number_of_images; ++i) {
    unsigned char label;

<<<<<<< f49c590708b62703a74ba695516b4a2f1e20dfbd
    ifs.read((char*)&label, sizeof(char));
    if (!ifs) {
      std::cerr << "Read label error!" << std::endl;
=======
  for (int i = 0; i < number_of_images; ++i) {
    char label;

    ifs.read(&label, 1);
    if (!ifs) {
      std::cerr << "File read error!" << std::endl;
>>>>>>> Change to use fstream.
      exit(1);
    }
    if (c_class == COARSE)
      labels[i] = OneHot<LabelType>((unsigned long)label);
<<<<<<< f49c590708b62703a74ba695516b4a2f1e20dfbd
    ifs.read((char*)&label, sizeof(char));
    if (!ifs) {
      std::cerr << "Read label error!" << std::endl;
=======
    ifs.read(&label, 1);
    if (!ifs) {
      std::cerr << "File read error!" << std::endl;
>>>>>>> Change to use fstream.
      exit(1);
    }
    if (c_class == FINE)
      labels[i] = OneHot<LabelType>((unsigned long)label);

<<<<<<< f49c590708b62703a74ba695516b4a2f1e20dfbd
    unsigned char image[image_size];
    ifs.read((char*)image, sizeof(image));
    if (!ifs) {
      std::cerr << "Read image error!" << std::endl;
      exit(1);
    }
    for (int j = 0; j < image_size; ++j) {
      images[i][j] = image[j] / 255.0;
    }
=======
    ifs.read(&images[i], image_size);
    if (!ifs) {
      std::cerr << "File read error!" << std::endl;
      exit(1);
    }
    std::for_each(images[i].begin(), images[i].end(),
                  [](typename ImageType::Type x) {x /= 255.0;});
>>>>>>> Change to use fstream.
  }
  std::cerr << "Success read Cifar100 " <<
    (st==TRAIN ? "Train" : "Test") << " data." << std::endl;
  return {images, labels};
}


template<typename ImageType, typename LabelType>
Dataset<ImageType, LabelType> ReadCifar10Data(Status st) {
  auto start = std::chrono::steady_clock::now();
  std::ifstream ifs((st == TRAIN ? kCifar10TrainDataFilePath :
                     kCifar10TestDataFilePath), std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "File open error!!" << std::endl;
    exit(EXIT_FAILURE);
  }

  int number_of_images = (st == TRAIN) ? 50000 : 10000;
  int image_size = 32 * 32 * 3;

  std::vector<ImageType> images(number_of_images);
  std::vector<LabelType> labels(number_of_images);

  for (int i = 0; i < number_of_images; ++i) {
    unsigned char label;
    ifs.read((char*)&label, sizeof(label));
    if (!ifs) {
      std::cerr << "Read label error!" << std::endl;
      exit(1);
    }
    labels[i] = OneHot<LabelType>((unsigned long)label);

    unsigned char image[image_size];
    ifs.read((char*)image, sizeof(image));
    if (!ifs) {
      std::cerr << "Read image error!" << std::endl;
      exit(1);
    }
    for (int j = 0; j < image_size; ++j) {
      images[i][j] = image[j] / 255.0;
    }
  }
  auto diff = std::chrono::steady_clock::now() - start;
  std::cerr << "Success read Cifar10 "
            << (st==TRAIN ? "Train" : "Test") << " data. ("
            << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()
            << " msec)"<< std::endl;
  return {images, labels};
}
