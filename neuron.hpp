#pragma once

#include <vector>

#include "misc.hpp"

template<class T>
struct Neuron {
	T activation = 0;
	T bias = 0;
	size_t connectionCount = 0;
	std::vector<T> weights;
	T preSigmoid = 0;

	Neuron() = default;

	Neuron(size_t const connectionCount)
	: bias(randFloat()), connectionCount(connectionCount) {
		if(connectionCount) {
			weights.resize(connectionCount);
			for(auto &w : weights)
				w = randFloat();
		}
	}

	~Neuron() = default;
};
