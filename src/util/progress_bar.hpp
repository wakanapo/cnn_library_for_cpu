#include <cstdio>
#include <iostream>

bool progressBar(int i, int n) {
  if (i < n) {
    std::string null_bar = "", fill_bar = "";
    for (int j = 0; j < 100; ++j) {
      null_bar += "░";
      fill_bar += "█";
    }
    
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
