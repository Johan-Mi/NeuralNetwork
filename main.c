#include <stdlib.h>
#include <math.h>
#include <stdarg.h>

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
} NeuralNet;

float sigmoid(float);
float sigmoidDerivative(float);
float randFloat(void);
Neuron makeNeuron(unsigned);
Layer makeLayer(unsigned, unsigned);
NeuralNet makeNeuralNet(unsigned, ...);
void deleteNeuron(Neuron*);
void deleteLayer(Layer*);
void deleteNeuralNet(NeuralNet*);



int main(void) {
	NeuralNet net = makeNeuralNet(4, 2, 3, 3, 1);

	deleteNeuralNet(&net);
	return EXIT_SUCCESS;
}



Neuron makeNeuron(unsigned connectionCount) {
	if(connectionCount) {
		Neuron ret = {0, randFloat(), connectionCount, MKARR(float, connectionCount)};
		for(unsigned i = 0; i < connectionCount; i++)
			ret.weights[i] = randFloat();
		return ret;
	} else {
		return (Neuron){0, 0, 0, NULL};
	}
}

Layer makeLayer(unsigned size, unsigned prevSize) {
	Layer ret = {size, MKARR(Neuron, size)};
	for(unsigned i = 0; i < size; i++)
		ret.neurons[i] = makeNeuron(prevSize);
	return ret;
}

NeuralNet makeNeuralNet(unsigned layerCount, ...) {
	NeuralNet ret = {layerCount, MKARR(Layer, layerCount)};
	va_list args;
	va_start(args, layerCount);
	for(unsigned i = 0; i < layerCount; i++)
		ret.layers[i] = makeLayer(va_arg(args, unsigned), i ? ret.layers[i - 1].size : 0);
	va_end(args);
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
}

float sigmoid(float x) {
	return 1 / (1 - exp(x));
}

float sigmoidDerivative(float x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

float randFloat(void) {
	return 2 * ((float)rand() / RAND_MAX) - 1;
}
