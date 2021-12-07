//
// Created by piotr on 21/11/2021.
//

#ifndef NEURAL_NETS_CPP_MATRIX_MATRIX_H_
#define NEURAL_NETS_CPP_MATRIX_MATRIX_H_
#include <cassert>
#include <string>
#include <vector>
namespace matrix {

struct Shape
{
  Shape(size_t height, size_t width);
  Shape() : width(0), height(0){};

  size_t height;
  size_t width;

  bool operator==(const Shape& rhs) const;
  bool operator!=(const Shape& rhs) const;
};

template<class T>
class Matrix {

 public:
  Matrix(){};
  Matrix(size_t height, size_t width);
  explicit Matrix(const std::vector<std::vector<T>>& data);

  Matrix(const Matrix&) = default;
  Matrix& operator=(const Matrix&) = default;

  void Transpose();

  /// access specific value by reference
  /// \param i the 'height' part
  /// \param j the 'width' part
  /// \return targeted value
  T& Get(int i, int j) { return data_[ToInt(i, j)]; };

  /// access specific value by reference
  /// \param i the 'height' part
  /// \param j the 'width' part
  /// \return targeted value
  T Get(int i, int j) const { return data_[ToInt(i, j)]; };

  /// access specific value by reference
  /// \param i the 'height' part
  /// \param j the 'width' part
  /// \return targeted value
  T& Get(size_t i, size_t j) { return data_[ToInt(i, j)]; };

  /// overrides every value in matrix and changes it to given val
  /// \param val the new state of every val in matrix
  void Fill(T val) {
	for (auto& number : data_) number = val;
  };

  /// overrides every value in matrix and changes it to 0
  void Clear() { Fill(T(0)); };

  size_t GetWidth() const;
  size_t GetHeight() const;

  const std::vector<T>& GetData() const;

  void Add(const T& other);
  void Add(const Matrix<T>& other);
  void Add(const std::vector<T>& other);

  void Sub(const Matrix<T>& other);

  void Mul(const T& other);
  void Mul(const Matrix<T>& other);
  void Mul(const std::vector<T>& other);

  bool operator==(const Matrix& rhs) const;
  bool operator!=(const Matrix& rhs) const;

  bool IsVector() const { return shape_.width == 1; }

  //  template <class T>
  std::vector<T>& RawData() { return data_; }

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
  Shape shape_;
  std::vector<T> data_;
};

template<class T>
size_t Matrix<T>::ToInt(int i, int j) const {
  assert(i >= 0 && j >= 0);
  assert(i < GetHeight() && j < GetWidth());
  return i * GetWidth() + j;
}

template<class T>
size_t Matrix<T>::ToInt(size_t i, size_t j) const {
  assert(i < GetHeight() && j < GetWidth());
  return i * GetWidth() + j;
}
template<class T>
size_t Matrix<T>::GetWidth() const {
  return shape_.width;
}
template<class T>
size_t Matrix<T>::GetHeight() const {
  return shape_.height;
}

template<class T>
const std::vector<T>& Matrix<T>::GetData() const {
  return data_;
}

template<class T>
Matrix<T>::Matrix(size_t height, size_t width) : shape_(height, width) {
  data_.reserve(width * height);
  for (int i = 0; i < GetHeight(); i++)
	for (int j = 0; j < GetWidth(); j++) data_.push_back(T(0));
}
template<class T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& data)
	: shape_(data.size(), data.begin()->size()) {
  shape_.width = data.begin()->size();

  for (auto& i : data)
	if (i.size() != GetWidth())
	  throw "passed data matrix must have correctly defined size";

  shape_.height = data.size();

  data_.reserve(GetWidth() * GetHeight());

  for (int i = 0; i < GetHeight(); i++)
	for (int j = 0; j < GetWidth(); j++) data_.push_back(data[i][j]);
}

template<class T>
static Matrix<T> Add(const Matrix<T>& matrix_a, const T& value) {
  Matrix<T> addition(matrix_a);
  for (int i = 0; i < matrix_a.GetHeight(); i++)
	for (int j = 0; j < matrix_a.GetWidth(); j++) addition.Get(i, j) += value;
  return addition;
}

template<class T>
void Matrix<T>::Add(const T& other) {
  for (int i = 0; i < GetHeight(); i++)
	for (int j = 0; j < GetWidth(); j++) Get(i, j) += other;
}

template<class T>
static Matrix<T> Add(const Matrix<T>& matrix_a, const Matrix<T>& matrix_b) {
  if (matrix_a.GetWidth() != matrix_b.GetWidth())
	throw "incorrect matrix shape";

  if (matrix_a.GetHeight() != matrix_b.GetHeight())
	throw "incorrect matrix shape";
  Matrix<T> addition(matrix_a);
  for (int i = 0; i < matrix_a.GetHeight(); i++)
	for (int j = 0; j < matrix_a.GetWidth(); j++)
	  addition.Get(i, j) += matrix_b.Get(i, j);
  return addition;
}
template<class T>
static Matrix<T> Sub(const Matrix<T>& matrix_a, const Matrix<T>& matrix_b) {

  if (matrix_a.GetWidth() != matrix_b.GetWidth())
	throw "incorrect matrix shape";

  if (matrix_a.GetHeight() != matrix_b.GetHeight())
	throw "incorrect matrix shape";

  Matrix<T> addition(matrix_a);
  for (int i = 0; i < matrix_a.GetHeight(); i++)
	for (int j = 0; j < matrix_a.GetWidth(); j++)
	  addition.Get(i, j) -= matrix_b.Get(i, j);
  return addition;
}

template<class T>
void Matrix<T>::Add(const Matrix<T>& other) {
  if (other.GetWidth() != GetWidth() || other.GetHeight() != GetHeight())
	throw "incorrect vector shape";

  for (int i = 0; i < GetHeight(); i++)
	for (int j = 0; j < GetWidth(); j++) Get(i, j) += other.Get(i, j);
}

template<class T>
void Matrix<T>::Sub(const Matrix<T>& other) {
  if (other.GetWidth() != GetWidth() || other.GetHeight() != GetHeight())
	throw "incorrect matrix shape";

  for (int i = 0; i < GetHeight(); i++)
	for (int j = 0; j < GetWidth(); j++) Get(i, j) -= other.Get(i, j);
}

template<class T>
void Matrix<T>::Add(const std::vector<T>& other) {
  if (GetWidth() != other.size() || GetHeight() != 1)
	throw "incorrect vector shape";

  for (int j = 0; j < GetWidth(); j++) Get(0, j) += other[j];
}

template<class T>
void Matrix<T>::Mul(const T& other) {
  for (int i = 0; i < GetHeight(); i++)
	for (int j = 0; j < GetWidth(); j++) Get(i, j) *= other;
}

template<class T>
void Matrix<T>::Mul(const Matrix<T>& other) {
  if (GetWidth() != other.GetHeight()) throw "incorrect matrix sizes";

  Matrix<T> multiplication(GetHeight(), other.GetWidth());

  for (int i = 0; i < multiplication.GetHeight(); i++)
	for (int j = 0; j < multiplication.GetWidth(); j++) {
	  for (int k = 0; k < GetWidth(); k++)
		multiplication.Get(i, j) += Get(i, k) * other.Get(k, j);
	}
  *this = multiplication;
}

///            1,3
/// 4,5,6    x 2,4  = 50,74
///            6,7
template<class T>
static std::vector<T> Mul(const std::vector<T>& other,
						  const Matrix<T>& matrix_a) {

  // The number of columns in the first matrix should be equal to the number of
  // rows in the second.
  if (matrix_a.GetHeight() != other.size()) throw "incorrect matrix sizes";

  std::vector<T> multiplication;
  for (int i = 0; i < matrix_a.GetWidth(); i++) multiplication.push_back(T(0));

  for (int j = 0; j < matrix_a.GetWidth(); j++)
	for (int i = 0; i < matrix_a.GetHeight(); i++) {
	  multiplication[j] += matrix_a.Get(i, j) * other[i];
	}

  return multiplication;
}
template<class T>
static Matrix<T> Mul(const Matrix<T>& matrix_a, const Matrix<T>& matrix_b) {
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
template<class T>
static Matrix<T> Mul(const Matrix<T>& matrix_a, const T& value) {

  Matrix<T> multiplication(matrix_a);

  for (int i = 0; i < multiplication.GetHeight(); i++)
	for (int j = 0; j < multiplication.GetWidth(); j++) {
	  multiplication.Get(i, j) *= value;
	}
  return multiplication;
}

template<class T>
void Matrix<T>::Mul(const std::vector<T>& other) {
  if (GetHeight() != other.size()) throw "incorrect matrix sizes";

  Matrix<T> multiplication(GetHeight(), 1);

  for (int i = 0; i < multiplication.GetHeight(); i++)
	for (int j = 0; j < multiplication.GetWidth(); j++) {
	  for (int k = 0; k < GetWidth(); k++)
		multiplication.Get(i, j) += Get(i, k) * other[i];
	}
  *this = multiplication;
}

template<class T>
bool Matrix<T>::operator==(const Matrix& rhs) const {
  return GetWidth() == rhs.GetWidth() && GetHeight() == rhs.GetHeight()
		 && data_ == rhs.data_;
}
template<class T>
bool Matrix<T>::operator!=(const Matrix& rhs) const {
  return !(rhs == *this);
}
template<class T>
void Matrix<T>::Transpose() {

  auto copy(*this);

  std::swap(shape_.width, shape_.height);
  for (int i = 0; i < GetHeight(); i++)
	for (int j = 0; j < GetWidth(); j++) { Get(i, j) = copy.Get(j, i); }
}

}// namespace matrix

#endif// NEURAL_NETS_CPP_MATRIX_MATRIX_H_
