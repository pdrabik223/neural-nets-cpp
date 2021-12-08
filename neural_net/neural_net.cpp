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
NeuralNet::FeedForward(const std::vector<double> &input) {

  input_values_ = matrix::Matrix<double>(input.size(), 1);
  input_values_.RawData() = input;

  matrix::Matrix<double> buffer(input.size(), 1);
  buffer.RawData() = input;

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
double NeuralNet::PropagateBackwards(const std::vector<double> &expected,
                                     double learning_rate) {

  matrix::Matrix<matrix::Matrix<double>> nabla_b(network_layers_.size(), 1);
  matrix::Matrix<matrix::Matrix<double>> nabla_w(network_layers_.size(), 1);

  matrix::Matrix<double> expected_matrix(matrix::ConvertToMatrix(expected));

  // output layer
  matrix::Matrix<double> delta = HadamardProduct(
      CostFunction(expected),
      Layer::ApplyDerivative(Nodes(-1), ActivationFunction(-2)));
  nabla_b.Get(-1, 0) = delta;

  nabla_w.Get(-1, 0) = Mul(delta, Transpose(Activations(-2)));

  // hidden layers
  const matrix::Matrix<double> kSp =
      Layer::ApplyDerivative(Nodes(-2), ActivationFunction(-2));

  delta = HadamardProduct(
      Mul(matrix::Transpose(
              network_layers_[network_layers_.size() - 1].GetWeights()),
          delta),
      kSp);
  nabla_b.Get(-2, 0) = delta;

  nabla_w.Get(-2, 0) = Mul(delta, Transpose(input_values_));


  // -------- apply changes -------
  network_layers_[network_layers_.size() - 1].GetWeights().Sub(
      matrix::Mul(nabla_w.Get(-1, 0), learning_rate));

  network_layers_[network_layers_.size() - 1].GetBiases() =
      Sub(network_layers_[network_layers_.size() - 1].GetBiases(),
          Mul(nabla_b.Get(-1, 0), learning_rate));

  network_layers_[network_layers_.size() - 2].GetWeights().Sub(
      matrix::Mul(nabla_w.Get(-2, 0), learning_rate));

  network_layers_[network_layers_.size() - 2].GetBiases() =
      Sub(network_layers_[network_layers_.size() - 2].GetBiases(),
          Mul(nabla_b.Get(-2, 0), learning_rate));

  double sum = 0.0;

  auto matrix_error = CostFunction(expected);
  for (int i = 0; i < matrix_error.GetHeight(); i++)
    for (int j = 0; j < matrix_error.GetWidth(); j++)

      sum += matrix_error.Get(i, j);
  return sum;
}

matrix::Matrix<double>
NeuralNet::CostFunction(const std::vector<double> &expected_output) const {

  matrix::Matrix<double> error(expected_output.size(), 1);

  for (int i = 0; i < error.GetHeight(); i++)
    for (int j = 0; j < error.GetWidth(); j++)
      error.Get(i, j) = Activations(-1).Get(i, j) - expected_output[i];

  return error;
}
