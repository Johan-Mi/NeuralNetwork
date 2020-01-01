#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

#include "net.h"
#include "misc.h"

NeuralNet makeNeuralNet(const unsigned layerCount, ...) {
	NeuralNet ret = {layerCount, MKARR(Layer, layerCount)};
	va_list args;
	va_start(args, layerCount);
	for(unsigned i = 0; i < layerCount; i++)
		ret.layers[i] = makeLayer(va_arg(args, unsigned), i ? ret.layers[i - 1].size : 0);
	va_end(args);
	ret.outputCount = ret.layers[ret.layerCount - 1].size;
	ret.output = MKARR(float, ret.outputCount);
	ret.error = MKARR(float, ret.outputCount);
	return ret;
}

void deleteNeuralNet(NeuralNet* net) {
	if(net->layers) {
		for(unsigned i = 0; i < net->layerCount; i++)
			deleteLayer(&net->layers[i]);
		free(net->layers);
	}
	free(net->output);
	free(net->error);
}

void think(NeuralNet* net, const float* input, const float* expectedOutput) {
	for(unsigned i = 0; i < net->layers[0].size; i++)
		net->layers[0].neurons[i].activation = input[i];

	for(unsigned i = 1; i < net->layerCount; i++) {
		for(unsigned j = 0; j < net->layers[i].size; j++) {
			float sum = net->layers[i].neurons[j].bias;
			for(unsigned k = 0; k < net->layers[i].neurons[j].connectionCount; k++)
				sum += net->layers[i - 1].neurons[k].activation * net->layers[i].neurons[j].weights[k];
			net->layers[i].neurons[j].activation = sigmoid(sum);
		}
	}

	net->cost = 0;
	for(unsigned i = 0; i < net->outputCount; i++) {
		net->output[i] = net->layers[net->layerCount - 1].neurons[i].activation;
		net->error[i] = expectedOutput[i] - net->output[i];
		net->cost += net->error[i] * net->error[i];
	}
	net->cost /= net->outputCount;

	putchar('[');
	for(unsigned i = 0; i < net->layers[0].size; i++) {
		if(i)
			putchar(' ');
		printf("%f", input[i]);
	}
	printf("] -> [");
	for(unsigned i = 0; i < net->outputCount; i++) {
		if(i)
			putchar(' ');
		printf("%f", net->output[i]);
	}
	printf("] Expected: [");
	for(unsigned i = 0; i < net->outputCount; i++) {
		if(i)
			putchar(' ');
		printf("%f", expectedOutput[i]);
	}
	printf("] Cost: %f\n", net->cost);
}
