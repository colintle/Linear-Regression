#include <stdexcept>
#include <stdio.h>

#include "linear_regression.h"
#include "operations/add.h"
#include "operations/multiply.h"
#include "operations/scale.h"
#include "operations/subtract.h"
#include "operations/transpose.h"

LinearRegression::LinearRegression(size_t input_dim, size_t output_dim,
                                   double learning_rate, int iterations)
    : learning_rate(learning_rate), iterations(iterations),
      weights(matrix_library::Tensor<double>({input_dim, output_dim})),
      bias(matrix_library::Tensor<double>({1, output_dim})) {}

const matrix_library::Tensor<double> &LinearRegression::get_weights() const {
  return weights;
}

const matrix_library::Tensor<double> &LinearRegression::get_bias() const {
  return bias;
}

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

  matrix_library::Tensor<double> ones_row({1, N});
  for (size_t i = 0; i < N; ++i) {
    ones_row[{0, i}] = 1.0;
  }

  for (int iter = 0; iter < iterations; ++iter) {
    //                        X       Weights  Bias
    // Predictions: (N, D) = (N, M) * (M, D) + (1, D)
    matrix_library::Tensor<double> predictions =
        matrix_library::operations::multiply(X, weights);
    matrix_library::Tensor<double> bias_tiled({N, D});
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < D; ++j) {
        bias_tiled[{i, j}] = bias[{0, j}];
      }
    }
    predictions = matrix_library::operations::add(predictions, bias_tiled);

    // diff = predictions - y  -> shape (N, D)
    matrix_library::Tensor<double> diff =
        matrix_library::operations::subtract(predictions, y);

    // weight_gradients = (2/N) * (X^T * diff) -> shapes: (M,N) * (N,D) = (M,D)
    matrix_library::Tensor<double> XT =
        matrix_library::operations::transpose(X);
    matrix_library::Tensor<double> weight_gradients =
        matrix_library::operations::multiply(XT, diff);
    weight_gradients = matrix_library::operations::scale(
        weight_gradients, 2.0 / static_cast<double>(N));

    // bias_gradients = (2/N) * (ones_row * diff) -> shapes: (1,N) * (N,D) =
    // (1,D)
    matrix_library::Tensor<double> bias_gradients =
        matrix_library::operations::multiply(ones_row, diff);
    bias_gradients = matrix_library::operations::scale(
        bias_gradients, 2.0 / static_cast<double>(N));

    // weights -= learning_rate * weight_gradients
    weights = matrix_library::operations::subtract(
        weights,
        matrix_library::operations::scale(weight_gradients, learning_rate));

    // bias -= learning_rate * bias_gradients
    bias = matrix_library::operations::subtract(
        bias, matrix_library::operations::scale(bias_gradients, learning_rate));
  }
}

matrix_library::Tensor<double>
LinearRegression::predict(const matrix_library::Tensor<double> &X) const {
  size_t N = X.shape()[0]; // number of samples
  size_t M = X.shape()[1]; // number of features

  if (M != weights.shape()[0]) {
    throw std::invalid_argument(
        "Input feature dimension does not match model weights.");
  }
  size_t D = weights.shape()[1];
  if (bias.shape()[0] != 1 || bias.shape()[1] != D) {
    throw std::invalid_argument("Bias tensor has incorrect shape.");
  }

  auto predictions = matrix_library::operations::multiply(X, weights);
  matrix_library::Tensor<double> bias_tiled({N, D});
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < D; ++j) {
      bias_tiled[{i, j}] = bias[{0, j}];
    }
  }
  return matrix_library::operations::add(predictions, bias_tiled);
}

LinearRegression::~LinearRegression() = default;
