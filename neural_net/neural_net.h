//
// Created by piotr on 19/11/2021.
//

#ifndef NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_
#define NEURAL_NETS_CPP_NEURTAL_NET_NEURALNET_H_

#include "layer.h"
#include <fstream>
#include <string>

struct Nabla {
  Nabla() : weights(), biases() {}
  Nabla(const matrix::Matrix<matrix::Matrix<double>> &weights,
        const matrix::Matrix<matrix::Matrix<double>> &biases)
      : weights(weights), biases(biases) {}
  matrix::Matrix<matrix::Matrix<double>> weights;
  matrix::Matrix<matrix::Matrix<double>> biases;

  void operator+=(const Nabla &other) {

    if (weights.GetHeight() == 0) {
      weights = other.weights;
      biases = other.biases;
    } else {
      for (int i = 0; i < weights.GetHeight(); i++)
        weights.Get(i).Add(other.weights.Get(i));
      for (int i = 0; i < biases.GetHeight(); i++)
        biases.Get(i).Add(other.biases.Get(i));
    }
  }
  void operator/=(const int value) {
    for (int i = 0; i < weights.GetHeight(); i++)
      weights.Get(i).Div(value);
    for (int i = 0; i < biases.GetHeight(); i++)
      biases.Get(i).Mul(value);
  }
};

class NeuralNet {

public:
  /// \param hidden_layer_sizes array of  uint numbers that represent
  /// consecutive layer sizes
  NeuralNet(size_t input_layer_size,
            const std::vector<size_t> &hidden_layer_sizes,
            size_t output_layer_size);

  NeuralNet(size_t input_layer_size, size_t output_layer_size);

  NeuralNet(const std::string &path) { LoadFromFile(path); }

  /// sets all weights and biases to random value ranging from 0 to 1
  void FillRandom();
  /// sets all weights and biases to specified value
  /// \param value witch all  weights will be set to
  void FillWeights(double value);

  /// sets all biases to specified value
  /// \param value to witch biases will be set to
  void FillBiases(double value);

  void Show();

  matrix::Matrix<double> FeedForward(const matrix::Matrix<double> &input);

  Nabla PropagateBackwards(const matrix::Matrix<double> &error);

  void Update(Nabla nabla, double learning_rate) {
    // -------- apply changes -------

    for (int l = 1; l <= network_layers_.size(); l++) {

      Weights(-l).Sub(Mul(nabla.weights.Get(-l), learning_rate));

      Biases(-l).Sub(Mul(nabla.biases.Get(-l), learning_rate));
    }
  }

  matrix::Matrix<double>
  CostFunction(const matrix::Matrix<double> &expected_output) const;
  matrix::Matrix<double>
  PowCostFunction(const matrix::Matrix<double> &expected_output) const;

  const matrix::Matrix<double> &Activations(PyId id) const {
    if (id.id == -network_layers_.size() - 1)
      return input_values_;
    return network_layers_[id.ConvertId(network_layers_.size())]
        .GetActivatedNodes();
  }

  const matrix::Matrix<double> &Nodes(PyId id) const {

    if (id.id == -network_layers_.size() - 1)
      return input_values_;
    return network_layers_[id.ConvertId(network_layers_.size())].GetNodes();
  }
  matrix::Matrix<double> &Weights(PyId id) {

    return network_layers_[id.ConvertId(network_layers_.size())].GetWeights();
  }
  matrix::Matrix<double> &Biases(PyId id) {

    return network_layers_[id.ConvertId(network_layers_.size())].GetBiases();
  }

  ActivationFunction &GetActivationFunction(PyId id) {

    return network_layers_[id.ConvertId(network_layers_.size())]
        .GetActivationFunction();
  }

  Layer &GetLayer(unsigned layer_id) { return network_layers_[layer_id]; }

  size_t LayersCount() const { return network_layers_.size(); }
  void SaveToFile(const std::string &file_path) {
    std::fstream file;
    file.open(file_path + ".txt", std::ios::out);
    file << input_layer_size_ << "\n";
    file << network_layers_.size() << "\n";

    for (int i = 0; i < network_layers_.size(); i++) {
      network_layers_[i].GetWeights().AppendToFile(file);
      file << "\n";
      network_layers_[i].GetBiases().AppendToFile(file);
      file << "\n";
      file << (int)network_layers_[i].GetActivationFunction();
      file << "\n";
    }
    file.close();
  }
  void LoadFromFile(const std::string &file_path) {
    std::fstream file;
    file.open(file_path + ".txt", std::ios::in);
    file >> input_layer_size_;
    input_values_ = matrix::Matrix<double>(input_layer_size_,1);
    int network_layers_count;
    file >> network_layers_count;
    for (int i = 0; i < network_layers_count; i++) {
      matrix::Matrix<double> weights;
      matrix::Matrix<double> biases;
      int activation_function;

      weights.ReadFromFile(file);
      biases.ReadFromFile(file);
      file >> activation_function;
      network_layers_.emplace_back(weights, biases,
                                   ActivationFunction(activation_function));
    }
  }

private:
protected:
  size_t input_layer_size_;

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
