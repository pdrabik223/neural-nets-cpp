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
  /// \param value witch all  weights will be set to
  void FillWeights(double value) {
    for (auto &hidden_layer : hidden_layers_) {
      hidden_layer.FillWeights(value);
    }
  }

  /// sets all biases to specified value
  /// \param value to witch biases will be set to
  void FillBiases(double value) {
    for (auto &hidden_layer : hidden_layers_) {
      hidden_layer.FillBiases(value);
    }
  };

  void Show() {

    std::cout << "input layer ";
    hidden_layers_[0].Show();
    std::cout << std::endl;

    for (int i = 1; i < hidden_layers_.size() - 1; i++) {
      std::cout << "hidden layer " << i;
      hidden_layers_[i].Show();
      std::cout << std::endl;
    }

    std::cout << "output layer ";
    hidden_layers_[hidden_layers_.size() - 1].Show();
    std::cout << std::endl;
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
