#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "net.h"

int main(void) {
	srand(time(NULL));

	NeuralNet net = makeNeuralNet(2, 2, 1);
	
	T inputs[] = {
		0.f, 0.f,
		0.f, 1.f,
		1.f, 0.f,
		1.f, 1.f,
	};

	T outputs[] = {
		0.f,
		1.f,
		1.f,
		0.f,
	};

	for(unsigned i = 0; i < 3; i++)
		train(&net, 4, inputs, outputs);

	deleteNeuralNet(&net);

	return EXIT_SUCCESS;
}
