#pragma once

#include <cmath>

constexpr float sigmoid(float const x) {
	return 1 / (1 + std::exp(-x));
}

constexpr float sigmoidDerivative(float const x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

float randFloat(void);
