//
// Created by piotr on 19/11/2021.
//

#ifndef NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_
#define NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_

#include "layer.h"
class NeuralNet {
public:
  NeuralNet(size_t input_layer_size,
            const std::vector<size_t> &hidden_layer_sizes,
            size_t output_layer_size);

  /// sets all weights and biases to random value ranging from 0 to 1
  void FillRandom() {
    for (auto &hidden_layer : hidden_layers_) {
      hidden_layer.FillRandom();
    }
  }
  /// sets all weights and biases to specified value
  /// \param value witch all biases and weights will be set to
  void Fill(double value) {
    for (auto &hidden_layer : hidden_layers_) {
      hidden_layer.Fill(value);
    }
  }

  std::vector<double> FeedForward(const std::vector<double> &input) {
    std::vector<double> buffer = input;

    for (auto &hidden_layer : hidden_layers_) {
      buffer = hidden_layer.FeedForward(buffer);
    }

    return buffer;
  }

protected:
  size_t input_layer_size_;
  size_t output_layer_size;
  std::vector<Layer> hidden_layers_;
};

#endif // NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_
