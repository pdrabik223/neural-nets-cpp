//
// Created by piotr on 19/11/2021.
//

#include "layer.h"
Layer::Layer(size_t previous_layer_height, size_t layer_height)
    : layer_height(layer_height), previous_layer_height(previous_layer_height),
      weights_(previous_layer_height, layer_height), biases_(layer_height) {}

size_t Layer::GetLayerHeight() const { return layer_height; }
size_t Layer::GetPreviousLayerHeight() const { return previous_layer_height; }
void Layer::FillRandom() {
  for (auto &b : biases_)
    b = rand() / RAND_MAX;
  for (int i = 0; i < weights_.GetHeight(); i++)
    for (int j = 0; j < weights_.GetWidth(); j++)
      weights_.Get(i, j) = rand() / RAND_MAX;
}
void Layer::Fill(double value) {
  for (auto &b : biases_)
    b = value;

  for (int i = 0; i < weights_.GetHeight(); i++)
    for (int j = 0; j < weights_.GetWidth(); j++)
      weights_.Get(i, j) = value;
}
