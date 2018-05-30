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
  template<typename Model>
  void executeEvaluation(Model model, Dataset<typename Model::InputType,
                         typename Model::OutputType> test);
private:
  std::vector<float> genom_list_;
  float evaluation_;
};

class GeneticAlgorithm {
public:
  static GeneticAlgorithm setup();
  std::vector<Genom> crossover(const Genom& parent) const;
  Genom mutation(const Genom& parent) const;
  void nextGenerationGeneCreate();
  void run(std::string filename);
  void save(std::string filepath);
  void print(int i);
private:
  GeneticAlgorithm(int genom_length, int genom_num, float cross_rate,
                   float mutation_rate, int max_generation)
    : genom_length_(genom_length), genom_num_(genom_num),
      cross_rate_(cross_rate), mutation_rate_(mutation_rate),
      max_generation_(max_generation) {};
  int genom_length_;
  int genom_num_;
  float cross_rate_;
  float mutation_rate_;
  int max_generation_;
  void moveGenoms(std::vector<Genom>&& genom);
  std::vector<Genom> genoms_;
};
