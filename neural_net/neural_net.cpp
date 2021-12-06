//
// Created by piotr on 19/11/2021.
//

#include "neural_net.h"

NeuralNet::NeuralNet(size_t input_layer_size,
                     const std::vector<size_t> &hidden_layer_sizes,
                     size_t output_layer_size)
    : input_layer_size_(input_layer_size),
      output_layer_size(output_layer_size) {

  network_layers_.emplace_back(input_layer_size, hidden_layer_sizes[0],
                               NormalizingFunction::SIGMOID);

  for (int i = 1; i < hidden_layer_sizes.size(); i++) {
    network_layers_.emplace_back(hidden_layer_sizes[i - 1],
                                 hidden_layer_sizes[i],
                                 NormalizingFunction::SIGMOID);
  }

  network_layers_.emplace_back(hidden_layer_sizes.back(), output_layer_size,
                               NormalizingFunction::SIGMOID);
}
NeuralNet::NeuralNet(size_t input_layer_size, size_t output_layer_size)
    : input_layer_size_(input_layer_size),
      output_layer_size(output_layer_size) {

  network_layers_.emplace_back(input_layer_size, output_layer_size,
                               NormalizingFunction::SIGMOID);
}

std::vector<double> NeuralNet::FeedForward(const std::vector<double> &input) {
  std::vector<double> buffer = input;
  input_values = input;
  for (auto &network_layer : network_layers_)
    buffer = network_layer.FeedForward(buffer);

  return buffer;
}
void NeuralNet::Show() {

  std::cout << "input layer size: " << input_layer_size_ << std::endl;

  for (int i = 0; i < network_layers_.size() - 1; i++) {
    network_layers_[i].Show();
    std::cout << std::endl;
  }

  std::cout << "output layer ";
  network_layers_.back().Show();
  std::cout << std::endl;
}
void NeuralNet::FillBiases(double value) {
  for (auto &hidden_layer : network_layers_) {
    hidden_layer.FillBiases(value);
  }
}
void NeuralNet::FillWeights(double value) {
  for (auto &hidden_layer : network_layers_) {
    hidden_layer.FillWeights(value);
  }
}
void NeuralNet::FillRandom() {
  for (auto &hidden_layer : network_layers_) {
    hidden_layer.FillRandom();
  }
}

std::vector<double>
NeuralNet::ApplySigmoidDerivative(const std::vector<double> &vector_a) {

  std::vector<double> solution;
  solution.reserve(vector_a.size());

  for (auto i : vector_a)
    solution.push_back(Layer::SigmoidDerivative(i));
  return solution;
}

double NeuralNet::PropagateBackwards(const std::vector<double> &expected,
                                     double learning_rate) {
  // output layer
  std::vector<double> delta = HadamardProduct(
      CostFunction(expected),
      ApplySigmoidDerivative(Activations(network_layers_.size() - 1)));

  const std::vector<double> nabla_b = delta;

  const matrix::Matrix<double> nabla_w =
      DotProduct(Transpose(delta), Activations(network_layers_.size() - 2));

  // hidden layers
  const std::vector<double> sp =
      ApplySigmoidDerivative(Activations(network_layers_.size() - 2));

  delta = HadamardProduct(
      sp,
      Mul(Transpose(network_layers_[network_layers_.size() - 2].GetWeights()),
          delta));
  //
  //  delta = HadamardProduct(delta, sp);

  const std::vector<double> hidden_layer_bias_error = delta;

  const matrix::Matrix<double> hidden_layer_weight_gradient =
      DotProduct(Transpose(delta), input_values);

  // -------- apply changes -------
  network_layers_[network_layers_.size() - 1].GetWeights().Sub(
      Transpose(matrix::Mul(nabla_w, learning_rate)));

  network_layers_[network_layers_.size() - 1].GetBiases() =
      Sub(network_layers_[network_layers_.size() - 1].GetBiases(),
          Mul(nabla_b, learning_rate));

  network_layers_[network_layers_.size() - 2].GetWeights().Sub(
      Transpose(matrix::Mul(hidden_layer_weight_gradient, learning_rate)));

  network_layers_[network_layers_.size() - 2].GetBiases() =
      Sub(network_layers_[network_layers_.size() - 2].GetBiases(),
          Mul(hidden_layer_bias_error, learning_rate));

  double sum = 0.0;
  for (auto i : CostFunction(expected))
    sum += i;
  return sum;
}

std::vector<double>
NeuralNet::CostFunction(const std::vector<double> &expected_output) const {

  std::vector<double> error;
  error.reserve(Activations(LayersCount() - 1).size());

  for (int i = 0; i < Activations(LayersCount() - 1).size(); i++)
    error.push_back(Activations(LayersCount() - 1)[i] - expected_output[i]);

  return error;
}
