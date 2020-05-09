#include <stdlib.h>
#include <cmath>

#include "misc.hpp"

float sigmoid(float const x) {
	return 1 / (1 + std::exp(-x));
}

float sigmoidDerivative(float const x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

float randFloat(void) {
	return 2 * (static_cast<float>(rand()) / RAND_MAX) - 1;
}
