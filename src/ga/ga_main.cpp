#include "ga/genom.hpp"

int main(int argc, char* argv[]) {
  GeneticAlgorithm ga(16, 100, 20, 0.05, 0.05, 50);
  if (argc != 2) {
    std::cout << "Usage: ./bin/ga filepath" << std::endl;
    abort();
  }
  ga.run(argv[1]);
}
