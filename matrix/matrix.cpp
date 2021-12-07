//
// Created by piotr on 21/11/2021.
//

#include "matrix.h"

bool matrix::Shape::operator==(const matrix::Shape &rhs) const {
  return width == rhs.width && height == rhs.height;
}
bool matrix::Shape::operator!=(const matrix::Shape &rhs) const {
  return !(rhs == *this);
}
matrix::Shape::Shape(size_t height, size_t width)
    :height(height),width(width) {}

