//
// Created by piotr on 10/12/2021.
//
#include "matrix_double.h"
#include "neural_net.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <matrix.h>

#include "TApplication.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2F.h"

#include "TRootCanvas.h"
#include <random>

// double Sum(matrix::Matrix<double> &target) {
//   double sum = 0.0;
//   for (auto i : target.RawData())
//     sum += abs(i);
//   return sum;
// }

struct TestCase {
  TestCase() : input(784, 1), label(10, 1) {}
  TestCase(const std::vector<int> &input_values, int label_val)
      : input(784, 1), label(10, 1) {
    for (int i = 0; i < 784; i++)
      input.Get(i) = (double)input_values[i] / 255.0;

    for (int i = 0; i < 10; i++)
      if (i == label_val)
        label.Get(i) = 1.0;
      else
        label.Get(i) = 0.0;
  }

  /// 784 values from 0 to 1
  matrix::Matrix<double> input;
  /// 10 values only one is 1
  matrix::Matrix<double> label;
};

void LoadTestCases(const std::string &csv_file_path,
                   std::vector<TestCase> &target, int no_test_cases = 0) {
  target.clear();
  std::ifstream file(csv_file_path);
  std::string line;
  std::getline(file, line);
  if (no_test_cases == 0)
    while (file.good()) {
      int label;
      file >> label;
      char coma;
      file >> coma;
      std::vector<int> pixels;
      for (int i = 0; i < 784; i++) {
        int pixel;
        file >> pixel;
        if (i < 783)
          file >> coma;
        pixels.push_back(pixel);
      }
      target.emplace_back(pixels, label);
    }
  else
    for (int t = 0; t < no_test_cases; t++) {
      int label;
      file >> label;
      char coma;
      file >> coma;
      std::vector<int> pixels;
      for (int i = 0; i < 784; i++) {
        int pixel;
        file >> pixel;
        if (i < 783)
          file >> coma;
        pixels.push_back(pixel);
      }
      target.emplace_back(pixels, label);
    }
  file.close();
}
void DisplayTestCase(const TestCase &image, TApplication &app) {

  int frame_width = 28;
  int frame_height = 28;

  auto c = new TCanvas("image_canvas", "NeuralNets", 10, 10, 784, 784);
  auto histo = new TH2F("h2", "test", frame_width, 0, frame_width, frame_height,
                        0, frame_height);

  histo->SetStats(false);

  histo->SetContour(255);

  for (auto x = 0; x < frame_width; x++)
    for (auto y = 0; y < frame_height; y++) {
      histo->Fill(y, x, image.input.Get((x * frame_width) + y) + 0.1);
    }

  histo->Draw("colz text");
  c->Draw();

  TRootCanvas *rc = (TRootCanvas *)c->GetCanvasImp();
  app.Run();
}
int main(int argc, char **argv) {
  //  / load training dataset
  std::vector<TestCase> train_data;
  printf("load train data...\t");
  LoadTestCases("../MNIST-try/mnist_train.csv", train_data, 60'000);
  printf("done \n");

  std::vector<TestCase> test_data;
  printf("load test data...\t");
  LoadTestCases("../MNIST-try/mnist_test.csv", test_data, 10'000);
  printf("done \n");

  printf("train data set size: %d", train_data.size());
  printf("test data set size: %d", test_data.size());

  TApplication app("app", &argc, argv);
  //  DisplayTestCase(test_data[12], app);

  auto mg = TGraph();

  NeuralNet nn(784, {16, 16}, 10);
  //  nn.ActivationFunction(-3) = ActivationFunction::SIGMOID;
  //  nn.ActivationFunction(-2) = ActivationFunction::SIGMOID;
  nn.ActivationFunction(-1) = ActivationFunction::SIGMOID;
  nn.FillRandom();

  double learning_rate = 0.1;

  const int kEpochs = 10;
  const int kBatchSize = 1000;
  const int kMiniBatchSize = 10;

  TestCase one;

  int k = 0;
  for (int e = 0; e < kEpochs; e++) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(train_data.begin(), train_data.end(),
                 std::default_random_engine(seed));

    for (int b = 0; b < train_data.size() - kMiniBatchSize;
         b += kMiniBatchSize) {

      double error_sum = 0;
      std::cout << "e: " << e << " b: " << b / kMiniBatchSize;

      for (int i = 0; i < kMiniBatchSize; i++) {
        Nabla nabla;
        nn.FeedForward(train_data[b + i].input);
        auto error = nn.PowCostFunction(train_data[b + i].label);
        nabla += nn.PropagateBackwards(error);
        error_sum += Sum(error);
        nn.Update(nabla, learning_rate);
      }
//      nabla /= kMiniBatchSize;

      mg.SetPoint(k, k, error_sum / (double)kMiniBatchSize);
      std::cout << " error: " << error_sum / (double)kMiniBatchSize<< "\n";
      k += 1;
    }
    //    learning_rate -= 0.099;
  }

  printf("id\tlabel\tnn approximation\n");
  printf("=====================train data==================\n");
  for (int i = 0; i < 10; i++) {

    //    printf("%d\n", i);
    std::cout << "label:    \t"
              << ToString(matrix::Transpose(train_data[i].label)) << "\n";
    std::cout << "net aprox:\t"
              << ToString(
                     matrix::Transpose(nn.FeedForward(train_data[i].input)))
              << "\n";
  }
  printf("==============test data============\n");
  for (int i = 0; i < 10; i++) {

    //    printf("%d\n", i);
    std::cout << "label:    \t"
              << ToString(matrix::Transpose(test_data[i].label)) << "\n";
    std::cout << "net aprox:\t"
              << ToString(matrix::Transpose(nn.FeedForward(test_data[i].input)))
              << "\n";
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