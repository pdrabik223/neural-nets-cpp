//
// Created by piotr on 19/11/2021.
//
#include "matrix_double.h"
#include "neural_net.h"
#include <iostream>
int main() {

  MatrixD first({{1, 2, 3}, {4, 5, 6}});
  MatrixD second({{4, 5, 6}, {7, 8, 9}});
  std::cout << ToString(first);
//
//  NeuralNet test(2, {3, 3}, 2);
//  test.FillRandom();
//  std::cout << ToString(test.FeedForward({3, 4}));

  return 0;
}