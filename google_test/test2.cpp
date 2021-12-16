//
// Created by piotr on 22/11/2021.
//
#include "matrix_vector_operations.h"
#include <gtest/gtest.h>
#include <linear_layer.h>
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
  std::vector<int> solution({7, 10});
  EXPECT_TRUE(solution == matrix::Mul(test2, test));
}

TEST(Matrix, Vector_Multiplication_2x3_3) {

  matrix::Matrix<int> test({{1, 2}, {3, 4}, {5, 6}});
  std::vector<int> test2({1, 2, 3});
  std::vector<int> solution({22, 28});

  EXPECT_TRUE(solution == matrix::Mul(test2, test));
}

TEST(Matrix, Vector_Multiplication_3x2_2) {

  std::vector<int> test2({1, 2});
  matrix::Matrix<int> test({{1, 2}, {3, 4}, {5, 6}});

  std::vector<int> solution({{5, 11, 17}});

  EXPECT_ANY_THROW(matrix::Mul(test2, test));
}

TEST(Vector, Vector_sum) {
  std::vector<double> test = {1, 2, 3, 4};
  std::vector<double> test2 = {4, 5, 6, 7};

  std::vector<double> test3 = {5, 7, 9, 11};

  EXPECT_TRUE(test3 == Add(test, test2));
}

TEST(MatrixMul, 1) {
  std::vector<double> test = {1, 2, 3};
  std::vector<double> test_2 = {4, 5, 6};
  matrix::Matrix<double> test3({{4, 5, 6.0}, {8, 10, 12}, {12.0, 15.0, 18.0}});

  EXPECT_TRUE(test3 == MatrixMul(test, test_2));
}

TEST(VectorMul, 1) {
  std::vector<double> test = {1, 2, 3};
  double test_2 = 0.5;
  std::vector<double> test3({0.5, 1, 1.5});

  EXPECT_TRUE(test3 == Mul(test, test_2));
}
TEST(VectorMul, 2) {
  std::vector<double> test = {1, 2, 3};
  double test_2 = 4;
  std::vector<double> test3({4, 8, 12});

  EXPECT_TRUE(test3 == Mul(test, test_2));
}

TEST(MatrixMul, 2) {
  matrix::Matrix<double> test({{1, 2, 3}, {1, 2, 3}});
  double test_2 = 0.5;
  matrix::Matrix<double> test3({{1 * test_2, 2 * test_2, 3 * test_2},
                                {1 * test_2, 2 * test_2, 3 * test_2}});

  EXPECT_TRUE(test3 == Mul(test, test_2));
}
TEST(MatrixMul, 3) {
  matrix::Matrix<double> test({{1, 2, 3}, {1, 2, 3}});
  double test_2 = 4;
  matrix::Matrix<double> test3({{1 * test_2, 2 * test_2, 3 * test_2},
                                {1 * test_2, 2 * test_2, 3 * test_2}});

  EXPECT_TRUE(test3 == Mul(test, test_2));
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

TEST(NN_etwork, Feed_Forward) {

//  NeuralNet test(1, {4}, 1);
//
//  test.GetLayer(0).GetWeights() = matrix::Matrix<double>({{1.5, 2, 2.5, 3}});
//  test.GetLayer(0).GetBiases() = std::vector<double>({0.5, 1, 1.5, 2});
//
//  test.GetLayer(1).GetWeights() =
//      matrix::Matrix<double>({{1.5}, {2}, {2.5}, {3}});
//  test.GetLayer(1).GetBiases() = std::vector<double>({0.5});
//
//  std::vector<double> input = {4};
//
//  std::vector<double> expected_hidden_values = {6.5, 9, 11.5, 14};
//
//  for (auto &i : expected_hidden_values)
//    i = Layer::Sigmoid(i);
//
//
//  test.FeedForward(input);
//
//  std::vector<double> expected_output = {0.9999249};


}
