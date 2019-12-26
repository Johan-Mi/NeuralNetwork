#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define MKARR(t,c) (malloc((c)*sizeof(t)))

typedef struct {
	float activation;
	float bias;
	unsigned connectionCount;
	float* weights;
} Neuron;

typedef struct {
	unsigned size;
	Neuron* neurons;
} Layer;

typedef struct {
	unsigned layerCount;
	Layer* layers;
	float* output;
	unsigned outputCount;
	float* error;
	float cost;
} NeuralNet;

float sigmoid(const float);
float sigmoidDerivative(const float);
float randFloat(void);
Neuron makeNeuron(const unsigned);
Layer makeLayer(const unsigned, const unsigned);
NeuralNet makeNeuralNet(const unsigned, ...);
void deleteNeuron(Neuron*);
void deleteLayer(Layer*);
void deleteNeuralNet(NeuralNet*);
void think(NeuralNet*, const float*, const float*);



int main(void) {
	srand(time(NULL));

	NeuralNet net = makeNeuralNet(4, 2, 3, 3, 1);

	float input[] = {1.f, 0.f};
	float expectedOutput[] = {1.f};
	think(&net, input, expectedOutput);

	deleteNeuralNet(&net);
	return EXIT_SUCCESS;
}



Neuron makeNeuron(const unsigned connectionCount) {
	if(connectionCount) {
		Neuron ret = {0, randFloat(), connectionCount, MKARR(float, connectionCount)};
		for(unsigned i = 0; i < connectionCount; i++)
			ret.weights[i] = randFloat();
		return ret;
	} else {
		return (Neuron){0, 0, 0, NULL};
	}
}

Layer makeLayer(const unsigned size, const unsigned prevSize) {
	Layer ret = {size, MKARR(Neuron, size)};
	for(unsigned i = 0; i < size; i++)
		ret.neurons[i] = makeNeuron(prevSize);
	return ret;
}

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

void deleteNeuron(Neuron* neuron) {
	if(neuron->weights)
		free(neuron->weights);
}

void deleteLayer(Layer* layer)  {
	if(layer->neurons) {
		for(unsigned i = 0; i < layer->size; i++)
			deleteNeuron(&layer->neurons[i]);
		free(layer->neurons);
	}
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

float sigmoid(const float x) {
	return 1 / (1 + expf(-x));
}

float sigmoidDerivative(const float x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

float randFloat(void) {
	return 2 * ((float)rand() / RAND_MAX) - 1;
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
