#include <stdlib.h>
#include <math.h>

#include "misc.h"

T sigmoid(const T x) {
	return 1 / (1 + expf(-x));
}

T sigmoidDerivative(const T x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

T randFloat(void) {
	return 2 * ((T)rand() / RAND_MAX) - 1;
}
