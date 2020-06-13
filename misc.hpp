#pragma once

#include <cmath>

template<class T>
constexpr T sigmoid(T const x) {
	// Various implementations available
	// return 1 / (1 + std::exp(-x));
	// return x / (1 + std::abs(x));
	return 0.5 + 0.5 * std::tanh(x);
}

template<class T>
constexpr T sigmoidDerivative(T const x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

template<class T>
T randNormalized() {
	return 2 * (static_cast<T>(rand()) / static_cast<T>(RAND_MAX)) - 1;
}
