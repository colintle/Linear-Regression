// Tests for LinearRegression
#include "linear_regression.h"
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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
