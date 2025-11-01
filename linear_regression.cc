#include <stdio.h>

#include "linear_regression.h"

LinearRegression::LinearRegression(size_t input_dim, size_t output_dim,
                                   double learning_rate, int iterations)
    : learning_rate(learning_rate), iterations(iterations),
      weights(matrix_library::Tensor<double>({input_dim, output_dim})),
      bias(matrix_library::Tensor<double>({1, output_dim})) {}

void LinearRegression::fit(const matrix_library::Tensor<double> &X,
                           const matrix_library::Tensor<double> &y) {
  size_t N = X.shape()[0]; // number of samples
  size_t M = X.shape()[1]; // number of features
  size_t D = y.shape()[1]; // number of output dimensions

  if (weights.shape()[0] != M || weights.shape()[1] != D) {
    throw std::invalid_argument("Weights tensor has incorrect shape.");
  }
  if (bias.shape()[0] != 1 || bias.shape()[1] != D) {
    throw std::invalid_argument("Bias tensor has incorrect shape.");
  }
}

LinearRegression::~LinearRegression() = default;
