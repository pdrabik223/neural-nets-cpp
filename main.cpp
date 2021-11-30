//
// Created by piotr on 19/11/2021.
//
#include "matrix_double.h"
#include "neural_net.h"
#include <iostream>
#include <matrix.h>

double LinearFunction(double x) { return 5 * x + 5; }

int main() {

  NeuralNet test(1, {3}, 1);
  test.FillRandom();

  for (int i = 0; i < 100; i++) {

    std::vector<double> input;
    input.push_back((double)rand() / (double)RAND_MAX);

    std::vector<double> target;
    target.push_back(LinearFunction(input[0]));

    auto nnError = test.NNError(test.FeedForward(input), target);
    test.PropagateBackwards(nnError, 0.1);
    std::cout << "i: " << i << "  error: " << nnError[0] << std::endl;
  }

  //

  return 0;
}