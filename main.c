#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "neuron.h"
#include "layer.h"
#include "net.h"
#include "misc.h"

int main(void) {
	srand(time(NULL));

	NeuralNet net = makeNeuralNet(4, 2, 3, 3, 1);

	T input[] = {1.f, 0.f};
	T expectedOutput[] = {1.f};
	think(&net, input, expectedOutput);

	deleteNeuralNet(&net);

	return EXIT_SUCCESS;
}
