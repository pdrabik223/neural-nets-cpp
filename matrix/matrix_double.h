//
// Created by piotr on 19/11/2021.
//

#ifndef NEURAL_NETS_CPP_MATRIX_MATRIX_DOUBLE_H_
#define NEURAL_NETS_CPP_MATRIX_MATRIX_DOUBLE_H_

#include <assert.h>
#include <string>
#include <vector>

class MatrixD {
public:
  MatrixD(size_t height, size_t width);
  MatrixD(const std::vector<std::vector<double>> &data);
  MatrixD(const MatrixD &) = default;
  MatrixD &operator=(const MatrixD &) = default;

  /// access specific value by reference
  /// \param i the 'height' part
  /// \param j the 'width' part
  /// \return targeted value
  double &Get(int i, int j) { return data_[ToInt(i, j)]; };

  /// access specific value by reference
  /// \param i the 'height' part
  /// \param j the 'width' part
  /// \return targeted value
  double Get(int i, int j) const { return data_[ToInt(i, j)]; };

  const std::vector<double> &GetData() const;
  /// access specific value by reference
  /// \param i the 'height' part
  /// \param j the 'width' part
  /// \return targeted value
  double &Get(size_t i, size_t j) { return data_[ToInt(i, j)]; };

  /// overrides every value in matrix and changes it to given val
  /// \param val the new state of every val in matrix
  void Fill(double val);



  /// overrides every value in matrix and changes it to 0
  void Clear();

  /// matrix addition
  MatrixD operator+(const MatrixD &other) const;

  /// matrix addition and assigment
  void operator+=(const MatrixD &other);

  /// matrix to scalar addition
  MatrixD operator+(const double &other) const;

  /// matrix to scalar addition and assigment
  void operator+=(const double &other);

  /// matrix subtraction
  MatrixD operator-(const MatrixD &other) const;

  /// matrix subtraction and assigment
  void operator-=(const MatrixD &other);

  /// matrix to scalar subtraction
  MatrixD operator-(const double &other) const;

  /// matrix subtraction and assigment
  void operator-=(const double &other);

  /// matrix multiplication
  MatrixD operator*(const MatrixD &other) const;

  /// matrix multiplication and assigment
  void operator*=(const MatrixD &other);

  /// matrix to scalar multiplication
  MatrixD operator*(const double &other) const;

  /// matrix multiplication and assigment
  void operator*=(const double &other);

  /// matrix to scalar division
  MatrixD operator/(const double &other) const;

  /// matrix division and assigment
  void operator/=(const double &other);

  bool operator==(const MatrixD &rhs) const;
  bool operator!=(const MatrixD &rhs) const;

  size_t GetWidth() const;
  size_t GetHeight() const;

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
  std::vector<double> data_;
};
static std::string ToString(const MatrixD &other) {
  std::string output = "[[";

  for (int i = 0; i < other.GetHeight(); i++) {
    for (int j = 0; j < other.GetWidth(); j++) {
      if (j < other.GetHeight() - 1)
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

static std::string ToString(const std::vector<double> &other) {
  std::string output = "[";

  for (int i = 0; i < other.size(); i++) {
    if (i < other.size() - 1)
      output += std::to_string(other[i]) + ", ";
    else
      output += std::to_string(other[i]) + "]";
  }

  output += ']';

  return output;
}
#endif // NEURAL_NETS_CPP_MATRIX_MATRIX_DOUBLE_H_
