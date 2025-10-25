#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "tensor.h"

// N is the number of samples
// M is the number of features
// K is the number of samples to predict
// D is the number of output dimensions
class LinearRegression
{
public:
    LinearRegression();

    // copy constructor
    LinearRegression(const LinearRegression &other) = delete;
    // copy assignment operator
    LinearRegression &operator=(const LinearRegression &other) = delete;
    ~LinearRegression();
    // X shape: (N, M)
    // y shape: (N, D)
    void fit(const matrix_library::Tensor<double> &X, const matrix_library::Tensor<double> &y);
    // X shape: (K, M)
    matrix_library::Tensor<double> predict(const matrix_library::Tensor<double> &X) const;

private:
    double learning_rate;
    int iterations;

    // weights shape: (M, D)
    matrix_library::Tensor<double> weights;
    // bias shape: (1, D)
    matrix_library::Tensor<double> bias;
};

#endif // LINEAR_REGRESSION_H