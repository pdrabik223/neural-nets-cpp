//
// Created by piotr on 19/11/2021.
//

#include "neural_net.h"
#include <iostream>
#include "matrix.h"

#include "TApplication.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TGraph.h"
#include "TH1.h"
#include "TRootCanvas.h"

double LinearFunction(double x, double y) { return 2 * x + y; }

struct TestCase {
  TestCase() : input(2, 1), label(2, 1) {
    input.Get(0) = (double)rand() / (double)RAND_MAX;
    input.Get(1) = (double)rand() / (double)RAND_MAX;
    label.Get(0) = LinearFunction(input.Get(0), input.Get(1));
    label.Get(1) = LinearFunction(input.Get(1), input.Get(0));
  }
  /// 784 values from 0 to 1
  matrix::Matrix<double> input;
  /// 10 values only one is 1
  matrix::Matrix<double> label;
};

int main(int argc, char **argv) {
  TApplication app("app", &argc, argv);
  auto learning_error = TGraph();

  NeuralNet neural_net(2, {2, 2}, 2);
  neural_net.FillRandom();

  double learning_rate = 0.1;

  const size_t kEpochs = 5;
  const size_t kBatchSize = 600;
  const size_t kMiniBach = 10;

  for (int b = 0; b < kEpochs; b++) {
    for (int i = 0; i < kBatchSize; i++) {

      double error_sum = 0;
      for (int k = 0; k < kMiniBach; k++) {

        TestCase test_case;
        Nabla nabla;

        neural_net.FeedForward(test_case.input);
        auto error = neural_net.CostFunction(test_case.label);
        nabla = neural_net.PropagateBackwards(error);
        neural_net.Update(nabla, learning_rate);

        error_sum += Sum(error);
        //        printf("epoch: %d\tbatch %d\terror %lf\n", b, i, error_sum);
      }
      error_sum /= (double)kMiniBach;
      learning_error.SetPoint(b * kBatchSize + i, b * kBatchSize + i,
                              abs(error_sum));

    }
  }
  printf("test no\t x value\ty value\tcorrect answer\t net estimation\n");
  neural_net.SaveToFile("../2paramNN");
  NeuralNet nn2("../2paramNN");
  for (int i = 0; i < 100; i++) {

    TestCase test_case;

    nn2.FeedForward(test_case.input);
    printf("%d\t%lf\t%lf\t%lf %lf\t%lf %lf\n", i,
           test_case.input.Get(0),
           test_case.input.Get(1),
           test_case.label.Get(0),
           test_case.label.Get(1),
           nn2.Activations(-1).Get(0),
           nn2.Activations(-1).Get(1));
  }

  auto c = new TCanvas("canvas", "NeuralNets", 10, 10, 800, 600);
  learning_error.SetTitle("Global_Net_Error;Iterations;Error");
  learning_error.SetMarkerStyle(22);
  learning_error.SetFillStyle(0);
  learning_error.SetMarkerSize(0);
  learning_error.SetDrawOption("LP");
  learning_error.SetLineColor(4);
  learning_error.SetLineWidth(2);
  learning_error.Draw();

  TRootCanvas *rc = (TRootCanvas *)c->GetCanvasImp();
  app.Run();

  return 0;
}
