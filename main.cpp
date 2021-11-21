//
// Created by piotr on 19/11/2021.
//
#include "matrix_double.h"
#include "neural_net.h"
#include <iostream>
int main() {

  //  NeuralNet test(2, {3, 3}, 2);
  //  test.Fill(1);
  //
  MatrixD test({1, 2, 3, 4, 5, 6});
  MatrixD test2({{std::vector<double>(1)}, {2}, {3}, {4}, {5}, {6}});
  std::cout << ToString(test * test2);

  return 0;
}