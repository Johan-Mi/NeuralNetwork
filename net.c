#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

#include "net.h"
#include "misc.h"

NeuralNet makeNeuralNet(const uint layerCount, ...) {
	NeuralNet ret = {layerCount, MKARR(Layer, layerCount)};
	va_list args;
	va_start(args, layerCount);
	for(uint i = 0; i < layerCount; i++)
		ret.layers[i] = makeLayer(va_arg(args, uint), i ? ret.layers[i - 1].size : 0);
	va_end(args);
	ret.outputCount = ret.layers[ret.layerCount - 1].size;
	ret.output = MKARR(T, ret.outputCount);
	ret.error = MKARR(T, ret.outputCount);
	return ret;
}

void deleteNeuralNet(NeuralNet* net) {
	if(net->layers) {
		for(uint i = 0; i < net->layerCount; i++)
			deleteLayer(&net->layers[i]);
		free(net->layers);
	}
	free(net->output);
	free(net->error);
}

void think(NeuralNet* net, const T* input, const T* expectedOutput) {
	for(uint i = 0; i < net->layers[0].size; i++)
		net->layers[0].neurons[i].activation = input[i];

	for(uint i = 1; i < net->layerCount; i++) {
		for(uint j = 0; j < net->layers[i].size; j++) {
			T sum = net->layers[i].neurons[j].bias;
			for(uint k = 0; k < net->layers[i].neurons[j].connectionCount; k++)
				sum += net->layers[i - 1].neurons[k].activation * net->layers[i].neurons[j].weights[k];
			net->layers[i].neurons[j].activation = sigmoid(sum);
		}
	}

	net->cost = 0;
	for(uint i = 0; i < net->outputCount; i++) {
		net->output[i] = net->layers[net->layerCount - 1].neurons[i].activation;
		net->error[i] = expectedOutput[i] - net->output[i];
		net->cost += net->error[i] * net->error[i];
	}
	net->cost /= net->outputCount;

	putchar('[');
	for(uint i = 0; i < net->layers[0].size; i++) {
		if(i)
			putchar(' ');
		printf("%f", input[i]);
	}
	printf("] -> [");
	for(uint i = 0; i < net->outputCount; i++) {
		if(i)
			putchar(' ');
		printf("%f", net->output[i]);
	}
	printf("] Expected: [");
	for(uint i = 0; i < net->outputCount; i++) {
		if(i)
			putchar(' ');
		printf("%f", expectedOutput[i]);
	}
	printf("] Cost: %f\n", net->cost);
}
