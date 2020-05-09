#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "layer.hpp"
#include "misc.hpp"

template<class T>
struct NeuralNet {
	size_t layerCount = 0;
	size_t outputCount = 0;
	T cost = 0;
	std::vector<Layer<T>> layers;
	std::vector<T> output;
	std::vector<T> error;
	std::unique_ptr<NeuralNet> delta;

	NeuralNet() = default;

	template<class... A>
	NeuralNet(A... layerSizes) {
		layerCount = sizeof...(layerSizes);
		layers.resize(layerCount);
		delta = std::make_unique<NeuralNet<T>>();
		delta->layers.resize(layerCount);

		size_t i = 0;
		([&]() mutable {
			layers[i] = Layer<T>(layerSizes, i ? layers[i - 1].size : 0);
			delta->layers[i] = Layer<T>(layerSizes, i ? layers[i - 1].size : 0);
			i++;
		}(), ...);

		outputCount = layers[layerCount - 1].size;
		output.resize(outputCount);
		error.resize(outputCount);
	}

	~NeuralNet() = default;

	void think(T const input[], T const expectedOutput[]) {
		for(size_t i = 0; i < layers[0].size; i++)
			layers[0].neurons[i].activation = input[i];

		for(size_t i = 1; i < layerCount; i++) {
			for(size_t j = 0; j < layers[i].size; j++) {
				T sum = layers[i].neurons[j].bias;
				for(size_t k = 0; k < layers[i].neurons[j].connectionCount; k++)
					sum += layers[i - 1].neurons[k].activation
						* layers[i].neurons[j].weights[k];
				layers[i].neurons[j].preSigmoid = sum;
				layers[i].neurons[j].activation = sigmoid(sum);
			}
		}

		cost = 0;
		for(size_t i = 0; i < outputCount; i++) {
			output[i] = layers[layerCount - 1].neurons[i].activation;
			error[i] = expectedOutput[i] - output[i];
			cost += error[i] * error[i];
		}
		cost /= outputCount;

		std::cout << '[';
		for(size_t i = 0; i < layers[0].size; i++)
			std::cout << input[i] << ' ';
		std::cout << "\b] -> [";
		for(size_t i = 0; i < outputCount; i++)
			std::cout << output[i] << ' ';
		std::cout << "\b] Expected: [";
		for(size_t i = 0; i < outputCount; i++)
			std::cout << expectedOutput[i] << ' ';
		std::cout << "\b] Cost: " << cost << '\n';
	}

	void train(size_t n, T const inputs[], T const expectedOutputs[]) {
		for(auto &a : delta->layers)
			for(auto &b : a.neurons)
				b.activation = 0;
		for(size_t i = 0; i < n; i++) {
			think(&inputs[i * layers[0].size],
					&expectedOutputs[i * outputCount]);
		}

	}
};
