#pragma once

#include <stdlib.h>

#define MKARR(t,c) (malloc((c)*sizeof(t)))

typedef unsigned int uint;
typedef float T;

T sigmoid(const T);
T sigmoidDerivative(const T);
T randFloat(void);
