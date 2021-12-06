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

double LinearFunction(double x) { return -20 * x; }

int main(int argc, char **argv) {
  TApplication app("app", &argc, argv);
  auto mg = new TGraph();

  NeuralNet test(1, {4}, 1);
  test.FillRandom();

  double learning_rate = 0.001;

  for (int i = 0; i < 1000; i++) {

    std::vector<double> input;
    input.push_back(((double)rand() / (double)RAND_MAX));

    std::vector<double> target;
    target.push_back(LinearFunction(input[0]));

    test.FeedForward(input);
    double error_sum = test.PropagateBackwards(target, learning_rate);

    std::cout << "i: " << i << "  error: " << error_sum << std::endl;
    mg->SetPoint(i, i, abs(error_sum));
  }

  auto *c = new TCanvas("canvas", "NeuralNets", 10, 10, 800, 600);
  mg->SetTitle("Global_Net_Error;Error;Iterations");
  mg->SetMarkerStyle(22);
  mg->SetFillStyle(0);
  mg->SetMarkerSize(0);
  mg->SetDrawOption("LP");
  mg->SetLineColor(4);
  mg->SetLineWidth(2);
  mg->Draw();

  TRootCanvas *rc = (TRootCanvas *)c->GetCanvasImp();
  app.Run();
  return 0;
}