//
// Created by piotr on 30/11/2021.
//

#ifndef NEURAL_NETS_CPP_MATRIX_MATRIX_VECTOR_OPERATIONS_H_
#define NEURAL_NETS_CPP_MATRIX_MATRIX_VECTOR_OPERATIONS_H_
#include "matrix.h"

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

template <class T> matrix::Matrix<T> Transpose(const matrix::Matrix<T> &other) {
  matrix::Matrix<T> transposed_matrix(other);
  transposed_matrix.Transpose();
  return transposed_matrix;
}

template <class T> matrix::Matrix<T> Transpose(const std::vector<T> &other) {
  matrix::Matrix<T> transposed_matrix(other.size(), 1);

  for (int i = 0; i < other.size(); i++)
    transposed_matrix.Get(i, 0) = other[i];

  return transposed_matrix;
}

template <class T>
static std::vector<T> Sub(const std::vector<T> &vector_a,
                          const std::vector<T> &vector_b) {

  if (vector_a.size() != vector_b.size())
    throw "incorrect vector shape";

  std::vector<T> sub(vector_a);
  for (auto i = 0; i < sub.size(); i++)
    sub[i] -= vector_b[i];

  return sub;
}
template <class T>
static std::vector<T> Mul(const std::vector<T> &vector_a, const T &value) {

  std::vector<T> output;
  for (auto i : vector_a)
    output.push_back(i * value);

  return output;
}

template <class T>
static std::vector<T> Mul(const matrix::Matrix<T> &matrix_a,
                          const std::vector<T> &vector_b) {

  if (matrix_a.GetWidth() != vector_b.size())
    throw "incorrect matrix sizes";

  std::vector<T> multiplication;

  for (int i = 0; i < matrix_a.GetHeight(); i++)
    multiplication.push_back(T(0));

  for (int j = 0; j < matrix_a.GetWidth(); j++)
    for (int i = 0; i < matrix_a.GetHeight(); i++) {
      multiplication[i] += matrix_a.Get(i, j) * vector_b[j];
    }

  return multiplication;
}

template <class T>
static matrix::Matrix<T> MatrixMul(const std::vector<T> &vector_a,
                                   const std::vector<T> &vector_b) {

  //  if (vector_a.size() != vector_b.size())
  //    throw "incorrect vector shape";

  matrix::Matrix<T> output(vector_a.size(), vector_b.size());

  for (int i = 0; i < vector_a.size(); i++)
    for (int j = 0; j < vector_b.size(); j++)
      output.Get(i, j) = vector_a[i] * vector_b[j];

  return output;
}

template <class T>
static matrix::Matrix<T> DotProduct(const matrix::Matrix<T> &matrix,
                                    const std::vector<T> vector) {

  if (matrix.GetWidth() != 1)
    throw "incorrect matrix size";

  matrix::Matrix<T> dot_product(matrix.GetHeight(), vector.size());
  dot_product.Fill(0);
  for (int i = 0; i < matrix.GetHeight(); i++)
    for (int j = 0; j < vector.size(); j++)
      dot_product.Get(i, j) += matrix.Get(i, 0) * vector[j];

  return dot_product;
}
template <class T>
static T SkalarDotProduct(const matrix::Matrix<T> &matrix,
                                    const std::vector<T> vector) {

  if(matrix.GetHeight() != 1)
    throw "incorrect matrix shape";

  if(matrix.GetWidth() != vector.size())
    throw "incorrect shape";

  T dot_product;

  for (int i = 0; i < matrix.GetWidth(); i++)
      dot_product += matrix.Get(0, 1) * vector[i];

  return dot_product;
}


template <class T>
static std::vector<T> HadamardProduct(const matrix::Matrix<T> &matrix_a,
                                      const std::vector<T> &vector_b) {

  if (matrix_a.GetHeight() != 1)
    throw "incorrect matrix dimensions";

  if (matrix_a.GetWidth() != vector_b.size())
    throw "incorrect vector dimensions";

  std::vector<T> hadamard_product(vector_b);
  for (auto i = 0; i < hadamard_product.size(); i++)
    hadamard_product[i] *= matrix_a.Get(0, i);

  return hadamard_product;
}

template <class T>
static std::vector<T> HadamardProduct(const std::vector<T> &vector_a,
                                      const std::vector<T> &vector_b) {

  if (vector_a.size() != vector_b.size())
    throw "incorrect vector dimensions";

  std::vector<T> hadamard_product(vector_b);
  for (auto i = 0; i < hadamard_product.size(); i++)
    hadamard_product[i] *= vector_a[i];

  return hadamard_product;
}

#endif // NEURAL_NETS_CPP_MATRIX_MATRIX_VECTOR_OPERATIONS_H_
