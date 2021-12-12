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
                               ActivationFunction::RELU);

  for (int i = 1; i < hidden_layer_sizes.size(); i++) {
    network_layers_.emplace_back(hidden_layer_sizes[i - 1],
                                 hidden_layer_sizes[i],
                                 ActivationFunction::RELU);
  }

  network_layers_.emplace_back(hidden_layer_sizes.back(), output_layer_size,
                               ActivationFunction::RELU);
}
NeuralNet::NeuralNet(size_t input_layer_size, size_t output_layer_size)
    : input_layer_size_(input_layer_size),
      output_layer_size(output_layer_size) {

  network_layers_.emplace_back(input_layer_size, output_layer_size,
                               ActivationFunction::RELU);
}

matrix::Matrix<double>
NeuralNet::FeedForward(const matrix::Matrix<double> &input) {

  input_values_ = input;

  matrix::Matrix<double> buffer(input);

  //  matrix::Matrix<double> input_matrix(input);

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

// There are only two hard things in Computer Science:
//  cache invalidation and naming things.
//
//                                      -- Phil Karlton
Nabla NeuralNet::PropagateBackwards(const matrix::Matrix<double> &error) {

  matrix::Matrix<matrix::Matrix<double>> nabla_b(network_layers_.size(), 1);
  matrix::Matrix<matrix::Matrix<double>> nabla_w(network_layers_.size(), 1);

  //  matrix::Matrix<double> expected_matrix(matrix::ConvertToMatrix(error));

  // output layer
  matrix::Matrix<double> delta = HadamardProduct(
      error, Layer::ApplyDerivative(Nodes(-1), ActivationFunction(-1)));
  nabla_b.Get(-1) = delta;

  nabla_w.Get(-1) = Mul(delta, Transpose(Activations(-2)));

  // hidden layers

  for (int l = 2; l <= network_layers_.size(); l++) {

    const matrix::Matrix<double> kSp =
        Layer::ApplyDerivative(Nodes(-l), ActivationFunction(-l));

    delta = HadamardProduct(Mul(matrix::Transpose(Weights(1 - l)), delta), kSp);

    nabla_b.Get(-l) = delta;

    nabla_w.Get(-l) = Mul(delta, matrix::Transpose(Activations(-1 - l)));
  }

  return {nabla_w, nabla_b};
}

matrix::Matrix<double>
NeuralNet::CostFunction(const matrix::Matrix<double> &expected_output) const {

  matrix::Matrix<double> error(expected_output.GetHeight(), 1);

  error = Sub(Activations(-1), expected_output);

  return error;
}
matrix::Matrix<double> NeuralNet::PowCostFunction(
    const matrix::Matrix<double> &expected_output) const {
  matrix::Matrix<double> error(expected_output.GetHeight(), 1);

  error = Sub(Activations(-1), expected_output);
error = matrix::HadamardProduct(error,error);
  return error;
}
