//
// Created by piotr on 21/11/2021.
//

#ifndef NEURAL_NETS_CPP_MATRIX_MATRIX_H_
#define NEURAL_NETS_CPP_MATRIX_MATRIX_H_
#include <assert.h>
#include <string>
#include <vector>
namespace matrix {
template <class T> class Matrix {

public:
  Matrix(size_t height, size_t width);
  explicit Matrix(const std::vector<std::vector<double>> &data);

  Matrix(const Matrix &) = default;
  Matrix &operator=(const Matrix &) = default;

  /// access specific value by reference
  /// \param i the 'height' part
  /// \param j the 'width' part
  /// \return targeted value
  T &Get(int i, int j) { return data_[ToInt(i, j)]; };

  /// access specific value by reference
  /// \param i the 'height' part
  /// \param j the 'width' part
  /// \return targeted value
  T Get(int i, int j) const { return data_[ToInt(i, j)]; };

  /// access specific value by reference
  /// \param i the 'height' part
  /// \param j the 'width' part
  /// \return targeted value
  T &Get(size_t i, size_t j) { return data_[ToInt(i, j)]; };

  /// overrides every value in matrix and changes it to given val
  /// \param val the new state of every val in matrix
  void Fill(T val) {
    for (auto &number : data_)
      number = val;
  };

  /// overrides every value in matrix and changes it to 0
  void Clear() { Fill(T(0)); };

  size_t GetWidth() const;
  size_t GetHeight() const;

  const std::vector<T> &GetData() const;

  void Add(const T &other);
  void Add(const Matrix<T> &other);
  void Add(const std::vector<T> &other);

  void Mul(const T &other);
  void Mul(const Matrix<T> &other);
  void Mul(const std::vector<T> &other);

  bool operator==(const Matrix &rhs) const;
  bool operator!=(const Matrix &rhs) const;

private:
  /// convert position in 2d space, to its 1D notation
  /// \param i the 'height' part
  /// \param j the 'width' part
  /// \return 1D conversion of given point
  size_t ToInt(int i, int j) const;

  /// convert position in 2d space, to its 1D notation
  /// \param i the 'height' part
  /// \param j the 'width' part
  /// \return 1D conversion of given point
  size_t ToInt(size_t i, size_t j) const;

protected:
  /// point a(0,0) is in left top corner, just like normal matrix
  /// width a.k.a. j part of a(i,j) symbol
  size_t width_;
  /// width a.k.a. i part of a(i,j) symbol
  size_t height_;
  std::vector<T> data_;
};

template <class T> size_t Matrix<T>::ToInt(int i, int j) const {
  assert(i >= 0 && j >= 0);
  assert(i < height_ && j < width_);
  return i * width_ + j;
}

template <class T> size_t Matrix<T>::ToInt(size_t i, size_t j) const {
  assert(i < height_ && j < width_);
  return i * width_ + j;
}
template <class T> size_t Matrix<T>::GetWidth() const { return width_; }
template <class T> size_t Matrix<T>::GetHeight() const { return height_; }

template <class T> const std::vector<T> &Matrix<T>::GetData() const {
  return data_;
}

template <class T>
Matrix<T>::Matrix(size_t height, size_t width)
    : width_(width), height_(height) {
  data_.reserve(width * height);
  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      data_.push_back(T(0));
}
template <class T>
Matrix<T>::Matrix(const std::vector<std::vector<double>> &data) {
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

template <class T>
static Matrix<T> Add(const Matrix<T> &matrix_a, const T &value) {
  Matrix<T> addition(matrix_a);
  for (int i = 0; i < matrix_a.GetHeight(); i++)
    for (int j = 0; j < matrix_a.GetWidth(); j++)
      addition.Get(i, j) += value;
  return addition;
}

template <class T> void Matrix<T>::Add(const T &other) {
  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      Get(i, j) += other;
}

template <class T>
static Matrix<T> Add(const Matrix<T> &matrix_a, const Matrix<T> &matrix_b) {
  if (matrix_a.GetWidth() != matrix_b.GetWidth())
    throw "incorrect matrix shape";
  if (matrix_a.Getheight() != matrix_b.GetHeight())
    throw "incorrect matrix shape";
  Matrix<T> addition(matrix_a);
  for (int i = 0; i < matrix_a.GetHeight(); i++)
    for (int j = 0; j < matrix_a.GetWidth(); j++)
      addition.Get(i, j) += matrix_b.Get(i, j);
  return addition;
}

template <class T> void Matrix<T>::Add(const Matrix<T> &other) {
  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      Get(i, j) += other.Get(i, j);
}

template <class T> void Matrix<T>::Add(const std::vector<T> &other) {
  if (width_ != other.size() || height_ != 1)
    throw "incorrect vector shape";

  for (int j = 0; j < width_; j++)
    Get(0, j) += other[j];
}

template <class T> void Matrix<T>::Mul(const T &other) {
  for (int i = 0; i < height_; i++)
    for (int j = 0; j < width_; j++)
      Get(i, j) *= other;
}

template <class T> void Matrix<T>::Mul(const Matrix<T> &other) {
  if (width_ != other.height_)
    throw "incorrect matrix sizes";

  Matrix<T> multiplication(height_, other.width_);

  for (int i = 0; i < multiplication.height_; i++)
    for (int j = 0; j < multiplication.width_; j++) {
      for (int k = 0; k < width_; k++)
        multiplication.Get(i, j) += Get(i, k) * other.Get(k, j);
    }
  *this = multiplication;
}

template <class T>
static std::vector<T> Mul(const Matrix<T> matrix_a,
                          const std::vector<T> &other) {
  if (matrix_a.GetWidth() != other.size())
    throw "incorrect matrix sizes";

  std::vector<T> multiplication;
  for (int i = 0; i < matrix_a.GetHeight(); i++)
    multiplication.push_back(T(0));

  for (int i = 0; i < matrix_a.GetHeight(); i++)
    for (int j = 0; j < matrix_a.GetWidth(); j++) {
      multiplication[i] += matrix_a.Get(i, j) * other[j];
    }
  return multiplication;
}
template <class T>
static Matrix<T> Mul(const Matrix<T> &matrix_a, const Matrix<T> &matrix_b) {
  if (matrix_a.GetWidth() != matrix_b.GetHeight())
    throw "incorrect matrix sizes";

  Matrix<T> multiplication(matrix_a.GetHeight(), matrix_b.GetWidth());

  for (int i = 0; i < multiplication.GetHeight(); i++)
    for (int j = 0; j < multiplication.GetWidth(); j++) {
      for (int k = 0; k < matrix_a.GetWidth(); k++)
        multiplication.Get(i, j) += matrix_a.Get(i, k) * matrix_b.Get(k, j);
    }
  return multiplication;
}
template <class T>
static Matrix<T> Mul(const Matrix<T> &matrix_a, const T &value) {

  Matrix<T> multiplication(matrix_a.GetHeight(), matrix_a.GetWidth());

  for (int i = 0; i < multiplication.height_; i++)
    for (int j = 0; j < multiplication.width_; j++) {
      for (int k = 0; k < matrix_a.GetWidth(); k++)
        multiplication.Get(i, j) += matrix_a.Get(i, k) * matrix_a.Get(k, j);
    }
  return multiplication;
}

template <class T> void Matrix<T>::Mul(const std::vector<T> &other) {
  if (height_ != other.size())
    throw "incorrect matrix sizes";

  Matrix<T> multiplication(height_, 1);

  for (int i = 0; i < multiplication.height_; i++)
    for (int j = 0; j < multiplication.width_; j++) {
      for (int k = 0; k < width_; k++)
        multiplication.Get(i, j) += Get(i, k) * other[i];
    }
  *this = multiplication;
}

template <class T> bool Matrix<T>::operator==(const Matrix &rhs) const {
  return width_ == rhs.width_ && height_ == rhs.height_ && data_ == rhs.data_;
}
template <class T> bool Matrix<T>::operator!=(const Matrix &rhs) const {
  return !(rhs == *this);
}

} // namespace matrix
template <class T> static std::string ToString(const matrix::Matrix<T> &other) {
  std::string output = "[[";

  for (int i = 0; i < other.GetHeight(); i++) {
    for (int j = 0; j < other.GetWidth(); j++) {

      if (j < other.GetWidth() - 1)
        output += std::to_string(other.Get(i, j)) + ", ";
      else
        output += std::to_string(other.Get(i, j)) + "]";
    }
    if (i < other.GetHeight() - 1)
      output += ",\n[";
  }
  output += ']';

  return output;
}

template <class T> static std::string ToString(const std::vector<T> &other) {
  std::string output = "[";

  for (int j = 0; j < other.size(); j++)
    if (j < other.size() - 1)
      output += std::to_string(other[j]) + ", ";
    else
      output += std::to_string(other[j]);

  output += "]";

  return output;
}

template <class T>
static std::vector<T> Add(const std::vector<T> &vector_a,
                          const std::vector<T> &vector_b) {

  if (vector_a.size() != vector_b.size())
    throw "incorrect vector shape";

  std::vector<T> sum(vector_a);
  for (auto i = 0; i < sum.size(); i++)
    sum[i] += vector_b[i];

  return sum;
}

#endif // NEURAL_NETS_CPP_MATRIX_MATRIX_H_
