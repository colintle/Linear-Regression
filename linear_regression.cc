#include <stdio.h>

#include "linear_regression.h"

LinearRegression::LinearRegression()
    : learning_rate(0.01), iterations(1000) {}

LinearRegression::~LinearRegression() = default;
