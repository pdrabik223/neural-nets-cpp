//
// Created by piotr on 10/12/2021.
//
#include "matrix_double.h"
#include "neural_net.h"
#include <iostream>
#include <matrix.h>

#include "TApplication.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TGraph.h"
#include "TH1.h"
#include "TRootCanvas.h"

double Sum(matrix::Matrix<double> &target) {
  double sum = 0.0;
  for (auto i : target.RawData())
    sum += i;
  return sum;
}

struct TestCase {
  TestCase() : input(784, 1), label(10, 1) {}
  TestCase(const std::vector<int> &input_values, int label_val)
      : input(784, 1), label(10, 1) {
    for (int i = 0; i < 784; i++)
      input.Get(i) = input_values[i];

    for (int i = 0; i < 10; i++)
      if (i == label_val)
        label.Get(i) = 1;
      else
        label.Get(i) = 0;
  }
  /// 784 values from 0 to 1
  matrix::Matrix<double> input;
  /// 10 values only one is 1
  matrix::Matrix<double> label;
};

int main(int argc, char **argv) {
  TApplication app("app", &argc, argv);
  auto mg = TGraph();

  NeuralNet test(784, {18, 16}, 10);
  test.FillRandom();

  double learning_rate = 0.01;

  TestCase one;

  for (int i = 0; i < 6000; i++) {

    Nabla nabla;
    test.FeedForward(one.input);
    auto error = test.CostFunction(one.label);
    nabla = test.PropagateBackwards(error);
    test.Update(nabla, learning_rate);
    mg.SetPoint(i, i, abs(Sum(error)));
  }

  auto c = new TCanvas("canvas", "NeuralNets", 10, 10, 800, 600);
  mg.SetTitle("Global_Net_Error;Iterations;Error");
  mg.SetMarkerStyle(22);
  mg.SetFillStyle(0);
  mg.SetMarkerSize(0);
  mg.SetDrawOption("LP");
  mg.SetLineColor(4);
  mg.SetLineWidth(2);
  mg.Draw();

  TRootCanvas *rc = (TRootCanvas *)c->GetCanvasImp();
  app.Run();

  return 0;
}