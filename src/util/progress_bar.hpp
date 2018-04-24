#include <cstdio>
#include <iostream>

namespace {
  std::string makeBar(std::string b) {
    std::string bar;
    for (int i = 0; i < 100; ++i) {
      bar += b;
    }
    return bar;
  }
  std::string null_bar = makeBar("░");
  std::string fill_bar = makeBar("█");
}  // namespace

bool progressBar(int i, int n) {
  if (i < n) {
    int progress = ((i+1)*100)/n;
    printf("\r%d%%|", progress);
    std::cout << fill_bar.substr(0, 3*progress)
              << null_bar.substr(0, 3*(100 - progress));
    printf("| %d/%d", i+1, n);
    progress == 100 ? std::cout << std::endl : std::cout << std::flush;
    return true;
  }
  return false;
}
