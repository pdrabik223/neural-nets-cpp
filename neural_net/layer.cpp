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
matrix::Matrix<double>
Layer::ApplyReluDerivative(const matrix::Matrix<double> &vector_a) {

  if (!vector_a.IsVector())
    throw "incorrect vector shape";

  matrix::Matrix<double> solution(vector_a.GetHeight(), 1);

  for (int i = 0; i < vector_a.GetHeight(); i++)
    for (int j = 0; j < vector_a.GetWidth(); j++)
      solution.Get(i, j) = (Layer::ReluDerivative(vector_a.Get(i, j)));

  return solution;
}
matrix::Matrix<double>
Layer::ApplySigmoidDerivative(const matrix::Matrix<double> &vector_a) {

  if (!vector_a.IsVector())
    throw "incorrect vector shape";

  matrix::Matrix<double> solution(vector_a.GetHeight(), 1);

  for (int i = 0; i < vector_a.GetHeight(); i++)
    for (int j = 0; j < vector_a.GetWidth(); j++)
      solution.Get(i, j) = (Layer::SigmoidDerivative(vector_a.Get(i, j)));

  return solution;
}
matrix::Matrix<double>
Layer::ApplyDerivative(const matrix::Matrix<double> &vector_a,
                       ActivationFunction activation_function) {
  switch (activation_function) {

  case ActivationFunction::RELU:
    return ApplyReluDerivative(vector_a);

  case ActivationFunction::SIGMOID:
    return ApplySigmoidDerivative(vector_a);
  }
}
double Layer::ReluDerivative(double val) {
  if (val <= 0)
    return 0.0;
  else
    return 1.0;
}
double Layer::SigmoidDerivative(double val) {
  return Sigmoid(val) * (1 - Sigmoid(val));
}
double Layer::Sigmoid(double val) { return (1.0 - 1.0 / (1.0 + exp(val))); }

 ActivationFunction& Layer::GetActivationFunction()  {
  return activation_function_;
}
