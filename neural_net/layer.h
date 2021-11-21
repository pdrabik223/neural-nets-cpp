//
// Created by piotr on 19/11/2021.
//

#ifndef NEURAL_NETS_CPP_NEURAL_NET_LAYER_H_
#define NEURAL_NETS_CPP_NEURAL_NET_LAYER_H_
#include "../matrix/matrix_double.h"

class Layer {
public:
  Layer(size_t previous_layer_height, size_t layer_height);
  Layer(const Layer &other) = default;
  Layer &operator=(const Layer &other) = default;

  /// feed forward values
  std::vector<double> FeedForward(const std::vector<double> &input) {
    return (weights_ * MatrixD(input) + MatrixD(biases_)).GetData();
  };

  /// sets all weights and biases to random value ranging from 0 to 1
  void FillRandom();
  /// sets all weights and biases to specified value
  /// \param value witch all biases and weights will be set to
  void Fill(double value);

  size_t GetLayerHeight() const;
  size_t GetPreviousLayerHeight() const;

protected:
  size_t layer_height;
  size_t previous_layer_height;
  MatrixD weights_;
  std::vector<double> biases_;
};

#endif // NEURAL_NETS_CPP_NEURAL_NET_LAYER_H_
