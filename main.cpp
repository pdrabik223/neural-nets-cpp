//
// Created by piotr on 19/11/2021.
//
#include "matrix_double.h"
#include <iostream>
int main() {

  MatrixD first({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  first.Fill(6);
  std::cout << ToString(first);

  return 0;
}