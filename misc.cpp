#include <stdlib.h>
#include <cmath>

#include "misc.hpp"

float randFloat(void) {
	return 2 * (static_cast<float>(rand()) / RAND_MAX) - 1;
}
