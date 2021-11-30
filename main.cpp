//
// Created by piotr on 19/11/2021.
//
#include "matrix_double.h"
#include "neural_net.h"
#include <iostream>
#include <matrix.h>

double LinearFunction(double x) { return 5 * x + 5; }

int main() {

  NeuralNet test(1, {4}, 1);
  test.FillRandom();

  for (int i = 0; i < 100; i++) {
//    test.Show();
//    std::cout<<"\n";
//    system("pause");

    std::vector<double> input;
    input.push_back((double)rand() / (double)RAND_MAX);

    std::vector<double> target;
    target.push_back(LinearFunction(input[0]));

    auto nn_error = NeuralNet::NNError(test.FeedForward(input), target);
    test.PropagateBackwards(nn_error, 0.1);


    std::cout << "i: " << i << "  error: " << nn_error[0] << std::endl;
  }

  //

  return 0;
}