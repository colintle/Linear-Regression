#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "tensor.h"

class LinearRegression {
public:
    LinearRegression();

    // copy constructor
    LinearRegression(const LinearRegression& other) = delete;
    // copy assignment operator
    LinearRegression &operator=(const LinearRegression& other) = delete;
    ~LinearRegression();

private:
    double learning_rate;
    int iterations;
};

#endif // LINEAR_REGRESSION_H