#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include "cnn/hinton_cifar10.hpp"
#include "ga/genom.hpp"
#include "util/box_quant.hpp"
#include "util/color.hpp"
#include "util/flags.hpp"
#include "util/read_data.hpp"
#include "util/timer.hpp"
#include "ga/set_gene.hpp"
#include "protos/genom.pb.h"

using Model = HintonCifar10<float>;

template<typename Model>
void Genom::executeEvaluation(Model model, Dataset<typename Model::InputType,
                              typename Model::OutputType> test) {
  model.cast();

  int cnt = 0;
  for (int i = 0; i < 4096; ++i) {
    unsigned long y = model.predict(test.images[i]);
    if (OneHot<typename Model::OutputType>(y) == test.labels[i])
      ++cnt;
  }

  evaluation_ = (float)cnt / (float)4096;
}

GeneticAlgorithm GeneticAlgorithm::setup() {
  Gene::Genoms genes;
  std::fstream input(Options::GetFirstGenomFile(),
                     std::ios::in | std::ios::binary);
  if (!genes.ParseFromIstream(&input)) {
    std::cerr << "Cannot load first genoms." << std::endl;
    exit(1);
  }

  GeneticAlgorithm ga(genes.genoms(0).gene_size(), genes.genoms_size(),
                      Options::GetCrossRate(),Options::GetMutationRate(),
                      Options::GetMaxGeneration());
  std::cerr << coloringText("Gene Length: " +
                            std::to_string(genes.genoms(0).gene_size()), BLUE)
            << coloringText(", Genom Num: " +
                            std::to_string(genes.genoms_size()), BLUE)
            << std::endl;
  
  std::vector<Genom> genoms;
  for (int i = 0; i < genes.genoms_size(); ++i) {
    std::vector<float> gene;
    for (int j = 0; j < genes.genoms(i).gene_size(); ++j) {
      gene.push_back(genes.genoms(i).gene(j));
    }
    genoms.push_back({gene, genes.genoms(i).evaluation()});
  }
  ga.moveGenoms(std::move(genoms));
  return ga;
}

void GeneticAlgorithm::moveGenoms(std::vector<Genom>&& genoms) {
  genoms_ = std::move(genoms);
}

std::vector<Genom> GeneticAlgorithm::crossover(const Genom& parent) const {
  /*
    二点交叉を行う関数
  */
  int center = rand() % (genom_length_ - 2) + 1;
  int range = rand() % std::min(center, (genom_length_ - center));
  int spouse = rand() % (genom_num_ / 2);
  std::vector<float> genom_one = parent.getGenom();
  std::vector<float> genom_two = genoms_[spouse].getGenom();
  auto inc_itr = std::lower_bound(genom_two.begin(), genom_two.end(),
                                  genom_one[center]);
  auto dic_itr = inc_itr;
  dic_itr--;
  for (int i = 0; i < range; ++i) {
    if (inc_itr != genom_two.end()) {
      std::swap(*inc_itr, genom_one[center+i]);
      ++inc_itr;
    }
    if (i != 0 && dic_itr >= genom_two.begin()) {
      std::swap(*dic_itr, genom_one[center-i]);
      --dic_itr;
    }
  }
  std::sort(genom_one.begin(), genom_one.end());
  std::sort(genom_two.begin(), genom_two.end());
  return {{genom_one, 0}, {genom_two, 0}};
}

Genom GeneticAlgorithm::mutation(const Genom& parent) const {
  /*
    突然変異関数
  */
  std::random_device seed;
  std::mt19937 mt(seed());
  std::uniform_real_distribution<> rand(0.0, 1.0);
  std::vector<float> genes = parent.getGenom();
  
  for (int i = 0; i < genom_length_; ++i) {
    float left = (i == 0) ? genes[i] - 0.05 : genes[i-1];
    float right = (i == genom_length_ - 1) ? genes[i] + 0.05 : genes[i+1];
    std::uniform_real_distribution<> new_pos(left, right);
    genes[i] = new_pos(mt);
  }
  return {genes, 0};
}


void GeneticAlgorithm::nextGenerationGeneCreate() {
  /*
    世代交代処理を行う関数
  */
  std::sort(genoms_.begin(), genoms_.end(),
            [](const Genom& a, const Genom& b) {
              return  a.getEvaluation() >  b.getEvaluation();
            });
  
  std::random_device seed;
  std::mt19937 mt(seed());
  std::uniform_real_distribution<> rand(0.0, 1.0);
  std::vector<Genom> new_genoms;
  new_genoms.reserve(genom_num_);
  new_genoms.push_back(genoms_[0]);

  while ((int)new_genoms.size() < genom_num_) {
    int idx = randomGenomIndex();
    auto r = rand(mt);
    /* 選択 */
    if (r > mutation_rate_ + cross_rate_) {
      new_genoms.push_back(genoms_[idx]);
      continue;
    }

    /* 突然変異 */
    if (r < mutation_rate_) {
      new_genoms.push_back(mutation(genoms_[idx]));
      continue;
    }

    /* 交叉 */
    if ((int)new_genoms.size() <= genom_num_ - 2) {
      auto genoms = crossover(genoms_[idx]);
      std::copy(genoms.begin(), genoms.end(), std::back_inserter(new_genoms));
      continue;
    }
  }
  genoms_ = std::move(new_genoms);
}

int GeneticAlgorithm::randomGenomIndex() const{
  std::random_device seed;
  std::mt19937 mt(seed());
  std::uniform_real_distribution<> rand(0.0, 1.0);
  float r = rand(mt);
  for (int i = 0; i < genom_num_; ++i) {
    float ratio = genoms_[i].getEvaluation() / (average_ * genom_num_);
    if (r < ratio)
      return i;
    r -= ratio;
  }
  return std::rand() % genom_num_;
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
  average_ = sum / genom_num_;
  std::cout << "generation: " << i << std::endl;
  std::cout << "Min: " << min << std::endl;
  std::cout << "Max: " << max << std::endl;
  std::cout << "Ave: " << average_ << std::endl;
  std::cout << "-------------" << std::endl;
}

void GeneticAlgorithm::save(std::string filename) {
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

void calculateEvaluation(const Model& model, const Dataset<Model::InputType, Model::OutputType>& test, Genom* genom) {
  GlobalParams::setParams(genom->getGenom());
  auto m = std::make_unique<Model>(model);
  genom->executeEvaluation(*m, test);
}

void GeneticAlgorithm::run(std::string filepath) {
  Timer timer;
  Model model;
  auto test = model.readData(TEST);
  model.load();
  for (int i = 0; i < max_generation_; ++i) {
    timer.start();
    if (i != 0) {
      /* 次世代集団の作成 */
      std::cerr << "Creating next generation ..... ";
      nextGenerationGeneCreate();
      std::cerr << coloringText("OK!", GREEN) << std::endl;
    }
    
    std::vector<std::thread> threads;
    /* 各遺伝子の評価*/
    std::cerr << "Evaluating genoms ..... ";
    for (auto& genom: genoms_) {
      if (genom.getEvaluation() <= 0) {
        threads.push_back(std::thread(calculateEvaluation, std::ref(model),
                                      std::ref(test), &genom));
      }
    }
    for (std::thread& th : threads) {
      th.join();
    }
    std::cerr << coloringText("OK!", GREEN) << std::endl;

    std::cerr << "Saving generation data ..... ";
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << i;
    save(filepath+"/generation"+ss.str());
    std::cerr << coloringText("OK!", GREEN) << std::endl;
    print(i);
    timer.show(SEC, "");
  }
}

std::string timestamp() {
  std::time_t t = time(nullptr);
  const tm* lt = localtime(&t);
  std::stringstream ss;
  ss << lt->tm_year-100;
  ss << std::setw(2) << std::setfill('0') << lt->tm_mon+1;
  ss << std::setw(2) << std::setfill('0') << lt->tm_mday;
  return ss.str();
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: ./bin/ga first_genom_file  input_file" << std::endl;
    return 1;
  }
  Options::ParseCommandLine(argc, argv);
  GeneticAlgorithm ga = GeneticAlgorithm::setup();
  std::string filepath = "data/" + timestamp();
  mkdir(filepath.c_str(), 0777);
  ga.run(filepath);
}
