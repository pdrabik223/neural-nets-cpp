//
// Created by piotr on 19/11/2021.
//

#include "neural_net.h"

NeuralNet::NeuralNet(size_t input_layer_size,
                     const std::vector<size_t> &hidden_layer_sizes,
                     size_t output_layer_size)
    : input_layer_size_(input_layer_size),
      output_layer_size(output_layer_size) {

  for (int i = 1; i < hidden_layer_sizes.size(); i++) {
    hidden_layers_.emplace_back(hidden_layer_sizes[i - 1],
                                hidden_layer_sizes[i]);
    functions.push_back(NormalizingFunction::RELU);
  }

  hidden_layers_.emplace_back(hidden_layer_sizes.back(), output_layer_size);
  functions.push_back(NormalizingFunction::RELU);
}
NeuralNet::NeuralNet(size_t input_layer_size, size_t output_layer_size)
    : input_layer_size_(input_layer_size),
      output_layer_size(output_layer_size) {

  hidden_layers_.emplace_back(input_layer_size, output_layer_size);
  functions.push_back(NormalizingFunction::SIGMOID);
}
std::string NeuralNet::ToString(NeuralNet::NormalizingFunction func) {
  switch (func) {
  case NeuralNet::NormalizingFunction::RELU:
    return "Relu";
  case NormalizingFunction::SIGMOID:
    return "Sigmoid";

  }
}
std::vector<double> &NeuralNet::ApplyNormalizingFunction(
    std::vector<double> &target_vector,
    NeuralNet::NormalizingFunction function_type) {
  switch (function_type) {

  case NormalizingFunction::RELU:
    for (auto &target : target_vector)
      target = Relu(target);
    break;
  case NormalizingFunction::SIGMOID:
    for (auto &target : target_vector)
      target = Sigmoid(target);
    break;
  }
  return target_vector;
}
std::vector<double> NeuralNet::FeedForward(const std::vector<double> &input) {
  std::vector<double> buffer = input;

  for (int i = 0; i < hidden_layers_.size(); i++) {
    buffer = hidden_layers_[i].FeedForward(buffer);
    buffer = ApplyNormalizingFunction(buffer, functions[i]);
  }

  return buffer;
}
void NeuralNet::Show() {

  std::cout << "input layer size: " << input_layer_size_ << std::endl;

  for (int i = 0; i < hidden_layers_.size() - 1; i++) {
    std::cout << "hidden layer " << i << "normalizing_function"
              << ToString(functions[i]);
    hidden_layers_[i].Show();
    std::cout << std::endl;
  }

  std::cout << "output layer ";
  hidden_layers_[hidden_layers_.size() - 1].Show();
  std::cout << std::endl;
}
void NeuralNet::FillBiases(double value) {
  for (auto &hidden_layer : hidden_layers_) {
    hidden_layer.FillBiases(value);
  }
}
void NeuralNet::FillWeights(double value) {
  for (auto &hidden_layer : hidden_layers_) {
    hidden_layer.FillWeights(value);
  }
}
void NeuralNet::FillRandom() {
  for (auto &hidden_layer : hidden_layers_) {
    hidden_layer.FillRandom();
  }
}
double NeuralNet::Relu(double val) {
  if (val < 0)
    return 0;
  else
    return val;
}
