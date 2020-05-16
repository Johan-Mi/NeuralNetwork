#pragma once

#include <cmath>

template<class T>
constexpr T sigmoid(T const x) {
	// return 1 / (1 + std::exp(-x));
	return 1 / (1 + std::abs(x)); // Faster approximation
}

template<class T>
constexpr T sigmoidDerivative(T const x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

template<class T>
T randNormalized() {
	return 2 * (static_cast<T>(rand()) / static_cast<T>(RAND_MAX)) - 1;
}
