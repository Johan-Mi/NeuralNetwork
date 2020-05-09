#pragma once

#include <vector>
#include <iostream>

#include "neuron.hpp"

template<class T>
struct Layer {
	size_t size = 0;
	std::vector<Neuron<T>> neurons;

	Layer() = default;

	Layer(size_t const size, size_t const prevSize)
	: size(size) {
		neurons.resize(size);
		for(auto &n : neurons)
			n = Neuron<T>(prevSize);
	}

	~Layer() = default;
};
