//
// Created by piotr on 19/11/2021.
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

int LinearFunction(int x, int y) { return x ^ y; }

int main(int argc, char **argv) {
  TApplication app("app", &argc, argv);
  auto mg = TGraph();

  NeuralNet test(2, {2}, 1);
  test.FillRandom();

  double learning_rate = 0.01;

  for (int i = 0; i < 6000; i++) {

    std::vector<double> input;
    input.push_back(rand() % 2);
    input.push_back(rand() % 2);

    matrix::Matrix<double> input_mat(input.size(), 1);
    input_mat.RawData() = input;

    std::vector<double> target;
    target.push_back(LinearFunction(int(input[0]), int(input[1])));

    matrix::Matrix<double> target_mat(target.size(), 1);
    target_mat.RawData() = target;

    Nabla nabla;

    test.FeedForward(input_mat);
    auto error = test.CostFunction(target_mat);
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
