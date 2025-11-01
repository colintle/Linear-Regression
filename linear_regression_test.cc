// Tests for LinearRegression
#include "linear_regression.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace matrix_library;

static Tensor<double> makeTensor(const std::vector<size_t> &shape,
                                 double fill = 0.0) {
  Tensor<double> t(shape);
  std::vector<double> data(t.size(), fill);
  t.set_data(std::move(data));
  return t;
}

static Tensor<double> makeTensorFrom(const std::vector<size_t> &shape,
                                     const std::vector<double> &data) {
  Tensor<double> t(shape);
  t.set_data(data);
  return t;
}

TEST(LinearRegressionTest, FitDoesNotThrowOnMatchingShapes) {
  const size_t N = 5, M = 3, D = 2;
  LinearRegression model(M, D, 0.01, 100);

  Tensor<double> X = makeTensor({N, M}, 1.0);
  Tensor<double> y = makeTensor({N, D}, 2.0);

  EXPECT_NO_THROW(model.fit(X, y));
}

TEST(LinearRegressionTest, FitThrowsOnMismatchedFeatureDim) {
  const size_t N = 5, M_model = 3, M_data = 4, D = 2;
  LinearRegression model(M_model, D, 0.01, 100);

  Tensor<double> X = makeTensor({N, M_data}, 1.0);
  Tensor<double> y = makeTensor({N, D}, 2.0);

  EXPECT_THROW(model.fit(X, y), std::invalid_argument);
}

TEST(LinearRegressionTest, FitThrowsOnMismatchedOutputDim) {
  const size_t N = 5, M = 3, D_model = 2, D_data = 1;
  LinearRegression model(M, D_model, 0.01, 100);

  Tensor<double> X = makeTensor({N, M}, 1.0);
  Tensor<double> y = makeTensor({N, D_data}, 2.0);

  EXPECT_THROW(model.fit(X, y), std::invalid_argument);
}

TEST(LinearRegressionTest, PredictReturnsCorrectShape) {
  const size_t M = 3, D = 2, K = 4;
  LinearRegression model(M, D, 0.01, 1);

  Tensor<double> X = makeTensor({K, M}, 1.0);

  auto Y = model.predict(X);
  ASSERT_EQ(Y.shape().size(), 2u);
  EXPECT_EQ(Y.shape()[0], K);
  EXPECT_EQ(Y.shape()[1], D);
}

TEST(LinearRegressionTest, PredictThrowsOnMismatchedFeatureDim) {
  const size_t M_model = 3, D = 1, K = 2, M_data = 4;
  LinearRegression model(M_model, D, 0.01, 1);

  Tensor<double> X = makeTensor({K, M_data}, 0.0);

  EXPECT_THROW({ (void)model.predict(X); }, std::invalid_argument);
}

TEST(LinearRegressionTest, PredictAfterTrainingSimple1D) {
  const size_t N = 6, M = 1, D = 1;
  LinearRegression model(M, D, 0.01, 5000);

  std::vector<double> x_data;
  x_data.reserve(N * M);
  for (size_t i = 0; i < N; ++i) {
    x_data.push_back(static_cast<double>(i));
  }
  Tensor<double> X = makeTensorFrom({N, M}, x_data);

  std::vector<double> y_data;
  y_data.reserve(N * D);
  for (size_t i = 0; i < N; ++i) {
    y_data.push_back(2.0 * static_cast<double>(i) + 3.0);
  }
  Tensor<double> y = makeTensorFrom({N, D}, y_data);

  EXPECT_NO_THROW(model.fit(X, y));

  const size_t K = 3;
  Tensor<double> X_pred = makeTensorFrom({K, M}, {6.0, 7.0, 8.0});
  Tensor<double> y_pred = model.predict(X_pred);

  ASSERT_EQ(y_pred.shape()[0], K);
  ASSERT_EQ(y_pred.shape()[1], D);

  const double expected[] = {15.0, 17.0, 19.0};
  for (size_t i = 0; i < K; ++i) {
    EXPECT_NEAR((y_pred[{i, 0}]), (expected[i]), 1e-1) << "at row " << i;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
