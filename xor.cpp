//
// Created by piotr on 19/11/2021.
//
#include "matrix_double.h"
#include "neural_net.h"
#include <iostream>
#include <matrix.h>

double Sum(matrix::Matrix<double> &target) {
  double sum = 0.0;
  for (auto i : target.RawData())
    sum += i;
  return sum;
}

int LinearFunction(int x, int y) { return x / y; }

int main() {

  NeuralNet test(2, {2}, 1);
  test.FillRandom();

  double learning_rate = 0.01;

  for (int i = 0; i < 6000; i++) {

    std::vector<double> input;
    input.push_back(rand() % 2 + 1);
    input.push_back(rand() % 2 + 1);

    std::vector<double> target;
    target.push_back(LinearFunction(int(input[0]), int(input[1])));

    Nabla nabla;
    test.FeedForward(input);
    auto error = test.CostFunction(target);
    nabla = test.PropagateBackwards(error);
    test.Update(nabla, learning_rate);
    //    learning_rate *= 0.99;
    //    std::cout << "i: " << i << "  error: " << error_sum << std::endl;
  }

  // =====test range ====
  int v1, v2;
  printf("network test:");
  while (true) {
    std::cin >> v1;
    std::cin >> v2;
    if (v1 == 9)
      break;

    std::vector<double> input;
    input.push_back(v1);
    input.push_back(v2);

    printf("network approximation:");
    std::cout << ToString(test.FeedForward(input));
  }

  return 0;
}
