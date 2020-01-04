#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "net.h"

int main(void) {
	srand(time(NULL));

	NeuralNet net = makeNeuralNet(2, 2, 1);

	T input[] = {1.f, 0.f};
	T expectedOutput[] = {1.f};
	think(&net, input, expectedOutput);

	deleteNeuralNet(&net);

	return EXIT_SUCCESS;
}
