//
// Created by piotr on 19/11/2021.
//

#include "neural_net.h"

NeuralNet::NeuralNet(size_t input_layer_size,
                     const std::vector<size_t> &hidden_layer_sizes,
                     size_t output_layer_size)
    : input_layer_size_(input_layer_size),
      output_layer_size(output_layer_size) {

  if (hidden_layers_.empty()) {
    hidden_layers_.emplace_back(input_layer_size, output_layer_size);
  }

  hidden_layers_.reserve(hidden_layer_sizes.size() + 1);
  hidden_layers_.emplace_back(input_layer_size, hidden_layer_sizes.front());

  for (int i = 1; i < hidden_layer_sizes.size(); i++)
    hidden_layers_.emplace_back(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]);

  hidden_layers_.emplace_back(hidden_layer_sizes.back(), output_layer_size);
}
