#pragma once

#include "cnn/cnn.hpp"
#include "ga/first_genoms.hpp"
#include "util/read_data.hpp"

class Genom {
public:
  Genom(std::vector<float> genom_list, float evaluation):
    genom_list_(genom_list), evaluation_(evaluation) {
  };
  std::vector<float> getGenom() const { return genom_list_; };
  float getEvaluation() const { return evaluation_; };
  void setGenom(std::vector<float> genom_list) { genom_list_ = genom_list; };
  void executeEvaluation();
private:
  std::vector<float> genom_list_;
  float evaluation_;
};

class GeneticAlgorithm {
public:
  GeneticAlgorithm(int genom_length, int genom_num, int elite_num,
                   float individual_mutation, float genom_mutation, int max_generation)
    : genom_length_(genom_length), genom_num_(genom_num), elite_num_(elite_num),
      individual_mutation_(individual_mutation), genom_mutation_(genom_mutation),
      max_generation_(max_generation) {
    for (auto genom: range) {
      genoms_.push_back(Genom(genom, 0));
    }
  };
  std::vector<Genom> selectElite() const;
  std::vector<Genom> crossover(std::vector<Genom> parents) const;
  void nextGenerationGeneCreate(std::vector<Genom>& progenies);
  void mutation();
  void run();
  void save(std::string filepath);
  void print(int i);
private:
  int genom_length_;
  int genom_num_;
  int elite_num_;
  float individual_mutation_;
  float genom_mutation_;
  int max_generation_;
  std::vector<Genom> genoms_;
};
