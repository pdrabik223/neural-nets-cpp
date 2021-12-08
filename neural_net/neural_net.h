//
// Created by piotr on 19/11/2021.
//

#ifndef NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_
#define NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_

#include "layer.h"
#include <string>


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

  matrix::Matrix<double> FeedForward(const std::vector<double> &input);

  double PropagateBackwards(const std::vector<double> &expected,
                            double learning_rate);

  matrix::Matrix<double>
  CostFunction(const std::vector<double> &expected_output) const;

  const matrix::Matrix<double> &Activations(PyId id) const {
    return network_layers_[id.ConvertId(network_layers_.size())]
        .GetActivatedNodes();
  }
  const matrix::Matrix<double> &Nodes(PyId id) const {
    return network_layers_[id.ConvertId(network_layers_.size())].GetNodes();
  }

  const ActivationFunction &ActivationFunction(PyId id) const {
    return network_layers_[id.ConvertId(network_layers_.size())]
        .GetActivationFunction();
  }

  Layer &GetLayer(unsigned layer_id) { return network_layers_[layer_id]; }

  size_t LayersCount() const { return network_layers_.size(); }

private:
protected:
  size_t input_layer_size_;
  size_t output_layer_size;
  matrix::Matrix<double> input_values_;
  std::vector<Layer> network_layers_;
};
static std::string ToString(ActivationFunction func) {
  switch (func) {
  case ActivationFunction::RELU:
    return "Relu";
  case ActivationFunction::SIGMOID:
    return "Sigmoid";
  }
}
#endif // NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_
