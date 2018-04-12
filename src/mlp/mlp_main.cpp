#include "mlp/mlp.hpp"
#include "half.hpp"

using half_float::half;
int main() {
  MLP<half> mlp;
  mlp.run();
  return 0;
}
  
