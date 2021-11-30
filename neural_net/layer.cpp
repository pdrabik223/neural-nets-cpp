//
// Created by piotr on 19/11/2021.
//

#include "layer.h"
Layer::Layer(size_t previous_layer_height, size_t layer_height,
             NormalizingFunction activation_function)
    : layer_height(layer_height), previous_layer_height(previous_layer_height),
      weights_(previous_layer_height, layer_height), biases_(layer_height),
      activation_function_(activation_function) {}

size_t Layer::GetLayerHeight() const { return layer_height; }
size_t Layer::GetPreviousLayerHeight() const { return previous_layer_height; }

void Layer::FillRandom() {
  for (auto &b : biases_)
    b = (double)rand() / (double)RAND_MAX;
  for (int i = 0; i < weights_.GetHeight(); i++)
    for (int j = 0; j < weights_.GetWidth(); j++)
      weights_.Get(i, j) = (double)rand() / (double)RAND_MAX;
}
void Layer::FillWeights(double value) {

  for (int i = 0; i < weights_.GetHeight(); i++)
    for (int j = 0; j < weights_.GetWidth(); j++)
      weights_.Get(i, j) = value;
}
void Layer::FillBiases(double value) {
  for (auto &val : biases_)
    val = value;
}
const std::vector<double> &Layer::GetNodes() const { return nodes_; }
matrix::Matrix<double> &Layer::GetWeights()  { return weights_; }
void Layer::SetWeights(const matrix::Matrix<double> &weights) {
  weights_ = weights;
}
void Layer::SetBiases(const std::vector<double> &biases) { biases_ = biases; }
std::vector<double> &Layer::GetBiases() { return biases_; }

std::vector<double> &
Layer::ApplyNormalizingFunction(std::vector<double> &target_vector,
                                NormalizingFunction function_type) {
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
double Layer::Relu(double val) {
  if (val < 0)
    return 0;
  else
    return val;
}