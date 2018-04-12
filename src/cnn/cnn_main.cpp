#include <cstdio>

#include "util/flags.hpp"
#include "util/read_data.hpp"
#include "cnn/cnn.hpp"
// #include "util/types.hpp"
#include "half.hpp"

using half_float::half;
int main(int argc, char* argv[]) {
  Flags::ParseCommandLine(argc, argv);
  if (Flags::GetType() == HALF) {
    if (Flags::IsTrain())
      CNN<half>::run();
    else
      CNN<half>::inference();
  }
  else {
    if (Flags::IsTrain())
      CNN<float>::run();
    else
      CNN<float>::inference();
  }
  return 0;
}
  
