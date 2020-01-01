#include <stdlib.h>
#include <math.h>

#include "misc.h"

float sigmoid(const float x) {
	return 1 / (1 + expf(-x));
}

float sigmoidDerivative(const float x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

float randFloat(void) {
	return 2 * ((float)rand() / RAND_MAX) - 1;
}
