//
// Created by piotr on 22/11/2021.
//
#include "matrix_vector_operations.h"
#include <gtest/gtest.h>
TEST(Matrix, size_constructor_2x2) {
  matrix::Matrix<double> test(2, 2);

  EXPECT_TRUE(test.GetHeight() == 2);
  EXPECT_TRUE(test.GetWidth() == 2);
  for (int i = 0; i < test.GetHeight(); i++)
    for (int j = 0; j < test.GetWidth(); j++)
      EXPECT_TRUE(test.Get(i, j) == 0.0);
}

TEST(Matrix, size_constructor_3x2) {
  matrix::Matrix<double> test(3, 2);

  EXPECT_TRUE(test.GetHeight() == 3);
  EXPECT_TRUE(test.GetWidth() == 2);
  for (int i = 0; i < test.GetHeight(); i++)
    for (int j = 0; j < test.GetWidth(); j++)
      EXPECT_TRUE(test.Get(i, j) == 0.0);
}
TEST(Matrix, size_constructor_1x5) {
  matrix::Matrix<double> test(1, 5);

  EXPECT_TRUE(test.GetHeight() == 1);
  EXPECT_TRUE(test.GetWidth() == 5);
  for (int i = 0; i < test.GetHeight(); i++)
    for (int j = 0; j < test.GetWidth(); j++)
      EXPECT_TRUE(test.Get(i, j) == 0.0);
}

TEST(Matrix, vector_constructor_2x2) {
  matrix::Matrix<int> test({{1, 2}, {3, 4}});

  EXPECT_TRUE(test.GetHeight() == 2);
  EXPECT_TRUE(test.GetWidth() == 2);

  int k = 1;
  for (int i = 0; i < test.GetHeight(); i++)
    for (int j = 0; j < test.GetWidth(); j++)
      EXPECT_TRUE(test.Get(i, j) == k++);
}
TEST(Matrix, Vector_Constructor_3x2) {

  matrix::Matrix<int> test({{1, 2}, {3, 4}, {5, 6}});
  EXPECT_TRUE(test.GetHeight() == 3);
  EXPECT_TRUE(test.GetWidth() == 2);
  int k = 1;

  for (int i = 0; i < test.GetHeight(); i++)
    for (int j = 0; j < test.GetWidth(); j++)
      EXPECT_TRUE(test.Get(i, j) == k++);
}

TEST(Matrix, Vector_Constructor_1x5) {
  matrix::Matrix<int> test({{1, 2, 3, 4, 5}});
  EXPECT_TRUE(test.GetHeight() == 1);
  EXPECT_TRUE(test.GetWidth() == 5);

  int k = 1;
  for (int i = 0; i < test.GetHeight(); i++)
    for (int j = 0; j < test.GetWidth(); j++)
      EXPECT_TRUE(test.Get(i, j) == k++);
}

TEST(Matrix, Equality_Operator_1x5) {
  matrix::Matrix<int> test({{1, 2, 3, 4, 5}});
  matrix::Matrix<int> test2({{1, 2, 3, 4, 5}});
  EXPECT_TRUE(test == test2);
}

TEST(Matrix, Copy_Constructor_1x5) {
  matrix::Matrix<int> test({{1, 2, 3, 4, 5}});
  auto test3(test);
  EXPECT_TRUE(test3 == test);
}

TEST(Matrix, Multiplication_error_3x2) {

  matrix::Matrix<int> test({{1, 2}, {3, 4}, {5, 6}});
  matrix::Matrix<int> test2({{1, 2}, {3, 4}, {5, 6}});

  EXPECT_ANY_THROW(test = matrix::Mul(test, test2));
}

TEST(Matrix, Multiplication_succes_3x2) {

  matrix::Matrix<int> test({{1, 2}, {3, 4}, {5, 6}});
  matrix::Matrix<int> test2({{1, 2, 3}, {4, 5, 6}});
  matrix::Matrix<int> solution({{9, 12, 15}, {19, 26, 33}, {29, 40, 51}});
  EXPECT_TRUE(solution == matrix::Mul(test, test2));
}
TEST(Matrix, Vector_Multiplication_2x2_2) {

  matrix::Matrix<int> test({{1, 2}, {3, 4}});
  std::vector<int> test2({1, 2});
  std::vector<int> solution({5, 11});
  EXPECT_TRUE(solution == matrix::Mul(test, test2));
}

TEST(Matrix, Vector_Multiplication_2x3_3)  {

  matrix::Matrix<int> test({{1, 2, 3}, {4, 5, 6}});
  std::vector<int> test2({1, 2, 3});
  std::vector<int> solution({14, 32});

  EXPECT_TRUE(solution == matrix::Mul(test, test2));
}

TEST(Matrix, Vector_Multiplication_3x2_2) {

  matrix::Matrix<int> test({{1, 2}, {3, 4}, {5, 6}});
  std::vector<int> test2({1, 2});

  std::vector<int> solution({{5, 11, 17}});

  EXPECT_TRUE(solution == matrix::Mul(test, test2));
}

TEST(Vector, Vector_sum) {
  std::vector<double> test = {1, 2, 3, 4};
  std::vector<double> test2 = {4, 5, 6, 7};

  std::vector<double> test3 = {5, 7, 9, 11};

  EXPECT_TRUE(test3 == Add(test, test2));
}

