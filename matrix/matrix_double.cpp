//
// Created by piotr on 19/11/2021.
//

#include "matrix_double.h"
MatrixD::MatrixD(size_t width, size_t height) : width_(width), height_(height) {
  data_.reserve(width * height);
  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      data_.push_back(0.0);
}
size_t MatrixD::ToInt(int i, int j) const {
  assert(i >= 0 && j >= 0);
  assert(i < height_ && j < width_);
  return i * width_ + j;
}
size_t MatrixD::ToInt(size_t i, size_t j) const {
  assert(i < height_ && j < width_);
  return i * width_ + j;
}
void MatrixD::Fill(double val) {
  for (auto &number : data_)
    number = val;
}
void MatrixD::Clear() { Fill(0); }
MatrixD::MatrixD(const std::vector<std::vector<double>> &data) {
  width_ = data.begin()->size();
  for (auto &i : data)
    if (i.size() != width_)
      throw "passed data matrix must have correctly defined size";
  height_ = data.size();
  data_.reserve(width_ * height_);

  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      data_.push_back(data[i][j]);
}
MatrixD MatrixD::operator+(const MatrixD &other) const {
  MatrixD addition(*this);

  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      addition.Get(i, j) += other.Get(i, j);

  return addition;
}

void MatrixD::operator+=(const MatrixD &other) {
  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      Get(i, j) += other.Get(i, j);
}

MatrixD MatrixD::operator+(const double &other) const {
  MatrixD addition(*this);

  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      addition.Get(i, j) += other;

  return addition;
}
void MatrixD::operator+=(const double &other) {
  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      Get(i, j) += other;
}
MatrixD MatrixD::operator-(const MatrixD &other) const {
  MatrixD subtraction(*this);

  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      subtraction.Get(i, j) -= other.Get(i, j);

  return subtraction;
}
void MatrixD::operator-=(const MatrixD &other) {
  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      Get(i, j) -= other.Get(i, j);
}

MatrixD MatrixD::operator-(const double &other) const {
  MatrixD subtraction(*this);

  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      subtraction.Get(i, j) -= other;

  return subtraction;
}
void MatrixD::operator-=(const double &other) {
  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      Get(i, j) -= other;
}
MatrixD MatrixD::operator*(const double &other) const {
  MatrixD multiplication(*this);

  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      multiplication.Get(i, j) *= other;

  return multiplication;
}
void MatrixD::operator*=(const double &other) {
  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      Get(i, j) *= other;
}

MatrixD MatrixD::operator/(const double &other) const {
  MatrixD multiplication(*this);

  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      multiplication.Get(i, j) /= other;

  return multiplication;
}
void MatrixD::operator/=(const double &other) {
  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      Get(i, j) /= other;
}
bool MatrixD::operator==(const MatrixD &rhs) const {
  return width_ == rhs.width_ && height_ == rhs.height_ && data_ == rhs.data_;
}
bool MatrixD::operator!=(const MatrixD &rhs) const { return !(rhs == *this); }
size_t MatrixD::GetWidth() const { return width_; }
size_t MatrixD::GetHeight() const { return height_; }