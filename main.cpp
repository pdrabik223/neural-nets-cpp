//
// Created by piotr on 19/11/2021.
//
#include "matrix_double.h"
#include "neural_net.h"
#include <iostream>
int main() {

  NeuralNet test(2, {3, 3}, 2);
  test.Fill(1);
  std::cout << ToString(test.FeedForward({3, 4}));

  return 0;
}