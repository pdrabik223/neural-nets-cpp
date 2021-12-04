//
// Created by piotr on 19/11/2021.
//

#ifndef NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_
#define NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_

#include "layer.h"
#include <string>

// todo move sigmoid and relu functions to layer class
// todo create tests for all of the mul, div etc... functions

class NeuralNet {

public:
  /// \param hidden_layer_sizes array of  uint numbers that represent
  /// consecutive layer sizes
  NeuralNet(size_t input_layer_size,
            const std::vector<size_t> &hidden_layer_sizes,
            size_t output_layer_size);

  NeuralNet(size_t input_layer_size, size_t output_layer_size);

  /// sets all weights and biases to random value ranging from 0 to 1
  void FillRandom();
  /// sets all weights and biases to specified value
  /// \param value witch all  weights will be set to
  void FillWeights(double value);

  /// sets all biases to specified value
  /// \param value to witch biases will be set to
  void FillBiases(double value);

  void Show();

  std::vector<double> FeedForward(const std::vector<double> &input);
  double PropagateBackwards(const std::vector<double> &output_error,
                            double learning_rate);

  static double CostFunction(const std::vector<double> &n_n_output,
                             const std::vector<double> &expected_output);
  static std::vector<double>
  NNError(const std::vector<double> &n_n_output,
          const std::vector<double> &expected_output);

  const std::vector<double> &Activations(unsigned layer_id) const {
    return network_layers_[layer_id].GetNodes();
  }

  Layer &GetLayer(unsigned layer_id) { return network_layers_[layer_id]; }

private:
  std::vector<double>
  ApplySigmoidDerivative(const std::vector<double> &vector_a);

protected:
  size_t input_layer_size_;
  size_t output_layer_size;
  std::vector<double> input_values;
  std::vector<Layer> network_layers_;
};
static std::string ToString(NormalizingFunction func) {
  switch (func) {
  case NormalizingFunction::RELU:
    return "Relu";
  case NormalizingFunction::SIGMOID:
    return "Sigmoid";
  }
}
#endif // NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_
