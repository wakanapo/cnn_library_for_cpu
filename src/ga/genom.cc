#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "ga/genom.hpp"
#include "util/box_quant.hpp"
#include "util/flags.hpp"
#include "util/read_data.hpp"
#include "ga/set_gene.hpp"
#include "protos/genom.pb.h"


void Genom::executeEvaluation(Model model,
                              const Dataset<typename Model::InputType,
                              typename Model::OutputType>& test) {
  model.cast();
  int cnt = 0;
  for (int i = 0; i < 4096; ++i) {
    unsigned long y = model.predict(test.images[i]);
    if (OneHot<Model::OutputType>(y) == test.labels[i])
      ++cnt;
  }

  evaluation_ = (float)cnt / (float)4096;
}

std::vector<Genom> GeneticAlgorithm::selectElite() const {
  std::vector<Genom> elites = genoms_;
  std::sort(elites.begin(), elites.end(),
            [](const Genom& a, const Genom& b) {
              return  a.getEvaluation() >  b.getEvaluation();
            });
  return std::vector<Genom>(elites.begin(), elites.begin() + elite_num_);
}

std::vector<Genom> GeneticAlgorithm::crossover(std::vector<Genom> parents) const {
  /*
    二点交叉を行う関数
  */
  std::vector<Genom> genoms;
  int cross_one = rand() % genom_length_;
  int cross_second = rand() % (genom_length_ - cross_one) + cross_one;

  for (int i = 0; i < parents.size() - 1; ++i) {
    std::vector<float> genom_one = parents[i].getGenom();
    std::vector<float> genom_two = parents[i+1].getGenom();
    float offset = genom_one[cross_one-1] - genom_two[cross_one-1];
    float ratio = (genom_one[cross_second] - genom_one[cross_one-1]) /
      (genom_two[cross_second] - genom_two[cross_one-1]);
    for (int j = cross_one; j < cross_second; ++j) {
      std::swap(genom_one[j], genom_two[j]);
      genom_one[j] = (genom_one[j] - offset) * ratio;
      genom_two[j] = (genom_two[j] + offset) / ratio;
    }
    genoms.push_back(Genom(genom_one, 0));
    genoms.push_back(Genom(genom_two, 0));
  }
  return genoms;
}

void GeneticAlgorithm::nextGenerationGeneCreate(std::vector<Genom>& progenies) {
  /*
    世代交代処理を行う関数
  */
  std::sort(genoms_.begin(), genoms_.end(),
            [](const Genom& a, const Genom& b) {
              return  a.getEvaluation() <  b.getEvaluation();
            });
  for (int i = 0; i < progenies.size(); ++i) {
    genoms_[i] = progenies[i];
  }
}

void GeneticAlgorithm::mutation() {
  /*
    突然変異関数
  */
  for (auto& genom: genoms_) {
    std::random_device seed;
    std::mt19937 mt(seed());
    std::uniform_real_distribution<> rand(0.0, 1.0);
    if (individual_mutation_ < rand(mt))
      continue;
    std::vector<float> new_genom;
    float offset = 0.0;
    for (int i = 0; i < genom_length_; ++i) {
      int gene = genom.getGenom()[i];
      if (genom_mutation_ > rand(mt) / 100.0) {
        float random = (rand(mt) - 0.5) * 2;
        float diff;
        if (random > 0)
          diff = (i == genom.getGenom().size() -1) ? 0.1 : genom.getGenom()[i+1] - gene;
        else
          diff = (i == 0) ? 0.1 : gene - genom.getGenom()[i-1];
        offset = random * diff;
      }
      new_genom.push_back(gene + offset);
    }
    genom.setGenom(new_genom);
  }
}

void GeneticAlgorithm::print(int i) {
  float min = 1.0;
  float max = 0;
  float sum = 0;
  
  for (auto& genom: genoms_) {
    float evaluation = genom.getEvaluation();
    sum += evaluation;
    if (evaluation < min)
      min = evaluation;
    if (evaluation > max)
      max = evaluation;
  }

  std::cout << "世代: " << i << std::endl;
  std::cout << "Min: " << min << std::endl;
  std::cout << "Max: " << max << std::endl;
  std::cout << "Ave: " << sum / genom_num_ << std::endl;
  std::cout << "-------------" << std::endl;
}

void GeneticAlgorithm::save(std::string filename) {
  std::string home = getenv("HOME");
  Gene::Genoms gs;
  for (auto genom : genoms_) {
    Gene::Gene* g = gs.add_genoms();
    for (auto gene : genom.getGenom()) {
      g->mutable_gene()->Add(gene);
    }
    g->set_evaluation(genom.getEvaluation());
  }

  std::fstream output(filename+".pb",
                      std::ios::out | std::ios::trunc | std::ios::binary);
  if (!gs.SerializeToOstream(&output))
    std::cerr << "Failed to save genoms." << std::endl;
}

void GeneticAlgorithm::run(std::string filepath) {
  Model model;
  Dataset<typename Model::InputType, typename Model::OutputType> test
    = model.readData(TRAIN);
  model.load();
  for (int i = 0; progressBar(i, max_generation_); ++i) {
    auto start = std::chrono::system_clock::now();
    std::vector<std::thread> threads;
    /* 各遺伝子の評価*/
    for (auto& genom: genoms_) {
      if (genom.getEvaluation() <= 0) {
        threads.push_back(std::thread([&genom, &model, &test] {
              GlobalParams::setParams(genom.getGenom());
              genom.executeEvaluation(model, test);
            }));
      }
    }
    for (std::thread& th : threads) {
      th.join();
    }
    
    print(i);
    save(filepath+std::to_string(i));
    
    /* エリートの選出 */
    std::vector<Genom> elites = selectElite();
    /* エリート遺伝子を交叉させ、子孫を作る */
    std::vector<Genom> progenies = crossover(elites);
    /* 次世代集団の作成 */
    nextGenerationGeneCreate(progenies);
    /* 突然変異 */
    mutation();
    auto diff = std::chrono::system_clock::now() - start;
    std::cout << "Time: "
              << std::chrono::duration_cast<std::chrono::seconds>(diff).count()
              << " sec."
              << std::endl;
  }
}

int main(int argc, char* argv[]) {
  GeneticAlgorithm ga(16, 100, 20, 0.05, 0.05, 50);
  if (argc != 3) {
    std::cout << "Usage: ./bin/ga test filepath" << std::endl;
    return 1;
  }
  Options::ParseCommandLine(argc, argv);
  ga.run("hinton_ga");
}
