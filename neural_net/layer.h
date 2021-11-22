//
// Created by piotr on 19/11/2021.
//

#ifndef NEURAL_NETS_CPP_NEURAL_NET_LAYER_H_
#define NEURAL_NETS_CPP_NEURAL_NET_LAYER_H_
#include "../matrix/matrix.h"
#include <iostream>

class Layer {
public:
  Layer(size_t previous_layer_height, size_t layer_height);
  Layer(const Layer &other) = default;
  Layer &operator=(const Layer &other) = default;

  /// feed forward values
  std::vector<double> FeedForward(const std::vector<double> &input) {
    return Add(matrix::Mul(weights_, input), biases_);
  };

  /// sets all weights and biases to random value ranging from 0 to 1
  void FillRandom();

  /// sets all weights to specified value
  /// \param value to witch weights will be set to
  void FillWeights(double value);

  /// sets all biases to specified value
  /// \param value to witch biases will be set to
  void FillBiases(double value);

  void Show() {
    std::cout << " weights:\n"
              << ToString(weights_) << "\n biases:\n"
              << ToString(biases_) << std::endl;
  };

  size_t GetLayerHeight() const;
  size_t GetPreviousLayerHeight() const;

protected:
  size_t layer_height;
  size_t previous_layer_height;
  matrix::Matrix<double> weights_;
  std::vector<double> biases_;
};

#endif // NEURAL_NETS_CPP_NEURAL_NET_LAYER_H_
