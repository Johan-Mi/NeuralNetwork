#pragma once

#include <stdlib.h>

#define MKARR(t,c) (malloc((c)*sizeof(t)))

float sigmoid(const float);
float sigmoidDerivative(const float);
float randFloat(void);
