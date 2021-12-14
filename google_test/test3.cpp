//
// Created by piotr on 07/12/2021.
//
#include "matrix_vector_operations.h"
#include <gtest/gtest.h>
#include <layer.h>
#include <neural_net.h>

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
  matrix::Matrix<int> test({{1}, {2}, {3}, {4}, {5}});
  EXPECT_TRUE(test.GetHeight() == 5);
  EXPECT_TRUE(test.GetWidth() == 1);

  int k = 1;
  for (int i = 0; i < test.GetHeight(); i++)
    for (int j = 0; j < test.GetWidth(); j++)
      EXPECT_TRUE(test.Get(i, j) == k++);
}

TEST(Matrix, Equality_Operator_1x5) {
  matrix::Matrix<int> test({{1}, {2}, {3}, {4}, {5}});
  matrix::Matrix<int> test2({{1}, {2}, {3}, {4}, {5}});
  EXPECT_TRUE(test == test2);
}

TEST(Matrix, Copy_Constructor_1x5) {
  matrix::Matrix<int> test({{1}, {2}, {3}, {4}, {5}});
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

TEST(HadamardProduct, Int) {
  std::vector<int> test1 = {1, 2, 3, 4, 5, 6};
  std::vector<int> test2 = {1, 2, 3, 4, 5, 6};

  std::vector<int> test3 = {1 * 1, 2 * 2, 3 * 3, 4 * 4, 5 * 5, 6 * 6};
  EXPECT_TRUE(test3 == HadamardProduct(test1, test2));
}

TEST(HadamardProduct, Double) {
  std::vector<double> test1 = {1, 2, 3, 4, 5, 6};
  std::vector<double> test2 = {1, 2, 3, 4, 5, 6};

  std::vector<double> test3 = {1 * 1, 2 * 2, 3 * 3, 4 * 4, 5 * 5, 6 * 6};
  EXPECT_TRUE(test3 == HadamardProduct(test1, test2));
}

TEST(HadamardProduct, error_test) {
  std::vector<double> test1 = {1, 2, 3, 4, 5, 6};
  std::vector<double> test2 = {2, 3, 4, 5, 6};

  std::vector<double> test3 = {2 * 2, 3 * 3, 4 * 4, 5 * 5, 6 * 6};
  EXPECT_ANY_THROW(test3 == HadamardProduct(test1, test2));
}

TEST(Sigmoid, 0) { EXPECT_TRUE(Layer::Sigmoid(0.0) == 0.5); }

TEST(Sigmoid, 4) { EXPECT_TRUE(Layer::Sigmoid(4) >= 0.982); }
TEST(Sigmoid, nimus_4) { EXPECT_TRUE(Layer::Sigmoid(-4) <= 0.018); }

TEST(SigmoidDerivative, 0) {
  EXPECT_TRUE(Layer::SigmoidDerivative(0.0) == 0.25);
}

TEST(SigmoidDerivative, 4) {
  EXPECT_TRUE(Layer::SigmoidDerivative(4) <= 0.018);
}
TEST(SigmoidDerivative, nimus_4) {
  EXPECT_TRUE(Layer::SigmoidDerivative(-4) <= 0.018);
}
