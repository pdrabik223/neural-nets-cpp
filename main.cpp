//
// Created by piotr on 19/11/2021.
//
#include "matrix_double.h"
#include "neural_net.h"
#include <iostream>
#include <matrix.h>
int main() {

  NeuralNet test(2,  2);
  test.FillWeights(1);
  test.FillBiases(0);
  test.Show();
  std::cout << ToString(test.FeedForward({-1, 5}));

  //

  return 0;
}