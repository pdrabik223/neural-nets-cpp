//
// Created by piotr on 19/11/2021.
//

#ifndef NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_
#define NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_

#include "layer.h"
#include <string>

#define E 2.7'182'818'284
class NeuralNet {

public:
  enum class NormalizingFunction { RELU, SIGMOID };
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

  static double CostFunction(const std::vector<double> &n_n_output,
                             const std::vector<double> &expected_output) {
    if (n_n_output.size() != expected_output.size())
      throw "incorrect vector dimensions";

    double error = 0.0;
    for (int i = 0; i < n_n_output.size(); i++)
      error += pow(n_n_output[i] - expected_output[i], 2);

    return error;
  }
  static std::vector<double>
  NNError(const std::vector<double> &n_n_output,
          const std::vector<double> &expected_output) {
    std::vector<double> error;
    error.reserve(n_n_output.size());
    for (int i = 0; i < n_n_output.size(); i++)
      error.push_back(n_n_output[i] - expected_output[i]);

    return error;
  }

private:
  static std::vector<double> &
  ApplyNormalizingFunction(std::vector<double> &target_vector,
                           NormalizingFunction function_type);

  static std::string ToString(NeuralNet::NormalizingFunction func);

  static double Relu(double val);
  static double Sigmoid(double val) { return 1.0 - 1.0 / (1.0 + exp(val)); }
  static double SigmoidDerivative(double val) {
    return Sigmoid(val) * (1 - Sigmoid(val));
  }

protected:
  size_t input_layer_size_;
  size_t output_layer_size;
  std::vector<Layer> hidden_layers_;
  std::vector<NormalizingFunction> functions;
};
static std::string ToString(NeuralNet::NormalizingFunction func) {
  switch (func) {
  case NeuralNet::NormalizingFunction::RELU:
    return "Relu";
  case NeuralNet::NormalizingFunction::SIGMOID:
    return "Sigmoid";

  }
}
#endif // NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_
