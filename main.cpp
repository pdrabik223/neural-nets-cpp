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

  for (int i = 0; i < 10000; i++) {
    std::vector<double> nn_error{0};
    //    test.Show();
    //    std::cout<<"\n";
    //    system("pause");

    //    for (int j = 0; j < 1000; j++) {
    std::vector<double> input;
    input.push_back(((double)rand() / (double)RAND_MAX));

    std::vector<double> target;
    target.push_back(LinearFunction(input[0]));

    nn_error =
        //          Add(nn_error,
        NeuralNet::NNError(test.FeedForward(input), target);
    //);
    //    }
    //    for (auto &j : nn_error)
    //      j = j / 1000.0;

    test.PropagateBackwards(nn_error, 0.1);

    std::cout << "i: " << i << "  error: " << nn_error[0] << std::endl;
  }

  //

  return 0;
}