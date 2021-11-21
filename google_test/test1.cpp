//
// Created by piotr on 21/11/2021.
//
#include "matrix_double.h"
#include <gtest/gtest.h>

TEST(MatrixClass, Size_Constructor_2x2) {

  MatrixD test(2, 2);
  EXPECT_TRUE(test.GetHeight() == 2);
  EXPECT_TRUE(test.GetWidth() == 2);
  for (int i = 0; i < test.GetHeight(); i++)
    for (int j = 0; j < test.GetWidth(); j++)
      EXPECT_TRUE(test.Get(i, j) == 0.0);
}
TEST(MatrixClass, Size_Constructor_3x2) {

  MatrixD test(3, 2);
  EXPECT_TRUE(test.GetHeight() == 3);
  EXPECT_TRUE(test.GetWidth() == 2);
  for (int i = 0; i < test.GetHeight(); i++)
    for (int j = 0; j < test.GetWidth(); j++)
      EXPECT_TRUE(test.Get(i, j) == 0.0);
}
TEST(MatrixClass, Size_Constructor_1x5) {

  MatrixD test(1, 5);
  EXPECT_TRUE(test.GetHeight() == 1);
  EXPECT_TRUE(test.GetWidth() == 5);
  for (int i = 0; i < test.GetHeight(); i++)
    for (int j = 0; j < test.GetWidth(); j++)
      EXPECT_TRUE(test.Get(i, j) == 0.0);
}
TEST(MatrixClass, Vector_Constructor_2x2) {

  MatrixD test({{1, 2}, {3, 4}});
  EXPECT_TRUE(test.GetHeight() == 2);
  EXPECT_TRUE(test.GetWidth() == 2);

  int k = 1;
  for (int i = 0; i < test.GetHeight(); i++)
    for (int j = 0; j < test.GetWidth(); j++)
      EXPECT_TRUE(test.Get(i, j) == k++);
}

TEST(MatrixClass, Vector_Constructor_3x2) {

  MatrixD test({{1, 2}, {3, 4}, {5, 6}});
  EXPECT_TRUE(test.GetHeight() == 3);
  EXPECT_TRUE(test.GetWidth() == 2);
  int k = 1;

  for (int i = 0; i < test.GetHeight(); i++)
    for (int j = 0; j < test.GetWidth(); j++)
      EXPECT_TRUE(test.Get(i, j) == k++);
}

TEST(MatrixClass, Vector_Constructor_1x5) {
  MatrixD test({1, 2, 3, 4, 5});
  EXPECT_TRUE(test.GetHeight() == 1);
  EXPECT_TRUE(test.GetWidth() == 5);

  int k = 1;
  for (int i = 0; i < test.GetHeight(); i++)
    for (int j = 0; j < test.GetWidth(); j++)
      EXPECT_TRUE(test.Get(i, j) == k++);
}
TEST(MatrixClass, Equality_Operator_1x5) {
  MatrixD test({1, 2, 3, 4, 5});
  MatrixD test2({1, 2, 3, 4, 5});
  EXPECT_TRUE(test == test2);
}

TEST(MatrixClass, Copy_Constructor_1x5) {
  MatrixD test({1, 2, 3, 4, 5});
  auto test3(test);
  EXPECT_TRUE(test3 == test);
}

TEST(MatrixClass, Addidion_1x5) {

  MatrixD test({1, 2, 3, 4, 5});
  MatrixD test2({1, 1, 1, 1, 1});
  MatrixD test3 = test + test2;

  int k = 2;
  for (int i = 0; i < test3.GetHeight(); i++)
    for (int j = 0; j < test3.GetWidth(); j++)
      EXPECT_TRUE(test3.Get(i, j) == k++);
}

TEST(MatrixClass, Addidion_3x2) {

  MatrixD test({{1, 2}, {3, 4}, {5, 6}});
  MatrixD test2({{1, 2}, {3, 4}, {5, 6}});
  MatrixD test4({{2, 4}, {6, 8}, {10, 12}});

  EXPECT_TRUE(test + test2 == test4);
}

TEST(MatrixClass, Multiplication_error_3x2) {

  MatrixD test({{1, 2}, {3, 4}, {5, 6}});
  MatrixD test2({{1, 2}, {3, 4}, {5, 6}});

  EXPECT_ANY_THROW(test = test * test2);
}

TEST(MatrixClass, Multiplication_succes_3x2) {

  MatrixD test({{1, 2}, {3, 4}, {5, 6}});
  MatrixD test2({{1, 2, 3}, {4, 5, 6}});
  MatrixD solution({{9, 12, 15}, {19, 26, 33}, {29, 40, 51}});
  EXPECT_TRUE(solution == test * test2);
}
