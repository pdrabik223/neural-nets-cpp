//
// Created by piotr on 19/11/2021.
//

#include "layer.h"

Layer::Layer(size_t previous_layer_height, size_t layer_height,
             ActivationFunction activation_function)
    : layer_height_(layer_height),
      previous_layer_height_(previous_layer_height),
      weights_(layer_height, previous_layer_height), biases_(layer_height, 1),
      activation_function_(activation_function), nodes_(layer_height, 1) {}

size_t Layer::GetLayerHeight() const { return layer_height_; }
size_t Layer::GetPreviousLayerHeight() const { return previous_layer_height_; }

void Layer::FillRandom() {

  for (int i = 0; i < biases_.GetHeight(); i++)
    for (int j = 0; j < biases_.GetWidth(); j++)
      biases_.Get(i, j) = (double)rand() / (double)RAND_MAX;

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

  for (int i = 0; i < biases_.GetHeight(); i++)
    for (int j = 0; j < biases_.GetWidth(); j++)
      biases_.Get(i, j) = value;
}
const matrix::Matrix<double> &Layer::GetNodes() const { return nodes_; }

matrix::Matrix<double> &Layer::GetWeights() { return weights_; }

void Layer::SetWeights(const matrix::Matrix<double> &weights) {
  weights_ = weights;
}
void Layer::SetBiases(matrix::Matrix<double> &biases) { biases_ = biases; }

matrix::Matrix<double> &Layer::GetBiases() { return biases_; }

matrix::Matrix<double> &
Layer::ApplyActivationFunction(const matrix::Matrix<double> &target_vector,
                               ActivationFunction function_type) {


  if (!target_vector.IsVector())
    throw "incorrect shape";

  activated_nodes_ = target_vector;
  switch (function_type) {
  case ActivationFunction::RELU:
    for (int i = 0; i < target_vector.GetHeight(); i++)
      for (int j = 0; j < target_vector.GetWidth(); j++)
        activated_nodes_.Get(i, j) = Relu(target_vector.Get(i, j));

    break;
  case ActivationFunction::SIGMOID:
    for (int i = 0; i < target_vector.GetHeight(); i++)
      for (int j = 0; j < target_vector.GetWidth(); j++)
        activated_nodes_.Get(i, j) = Sigmoid(target_vector.Get(i, j));
    break;
  }
  return activated_nodes_;
}
double Layer::Relu(double val) {
  if (val <= 0)
    return 0;
  else
    return val;
}

const matrix::Matrix<double> &Layer::GetActivatedNodes() const {
  return activated_nodes_;
}
