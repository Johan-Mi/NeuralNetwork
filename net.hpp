#pragma once

#include <memory>
#include <vector>
#include <array>
#include <fmt/format.h>

#include "range.hpp"
#include "misc.hpp"

template<class T, size_t... LayerSizes>
struct NeuralNet {
	constexpr static std::array<size_t, sizeof...(LayerSizes)> layerSizes
		= {LayerSizes...};
	constexpr static size_t outputCount = layerSizes[layerSizes.size() - 1];
	T cost = 0;
	std::array<std::vector<T>, layerSizes.size()> activations;
	std::array<std::vector<T>, layerSizes.size()> biases;
	std::array<std::vector<std::vector<T>>, layerSizes.size()> weights;
	std::vector<T> const &output = activations[layerSizes.size() - 1];
	std::array<T, outputCount> error;
	std::unique_ptr<NeuralNet> delta;

	NeuralNet(char) {
		// Used to create the delta network
	}

	NeuralNet()
	: delta(std::make_unique<NeuralNet<T, LayerSizes...>>(0)) {
		for(size_t i : util::lang::indices(layerSizes)) {
			activations[i].resize(layerSizes[i]);
			biases[i].resize(layerSizes[i]);
			weights[i].resize(layerSizes[i]);
			delta->activations[i].resize(layerSizes[i]);
			delta->biases[i].resize(layerSizes[i]);
			if(i) {
				for(auto &a : weights[i]) {
					a.resize(layerSizes[i - 1]);
				}
			}
		}

		for(auto &a : weights) {
			for(auto &b : a) {
				std::generate(begin(b), end(b), randNormalized<T>);
			}
		}

		for(auto &a : biases) {
			std::fill(begin(a), end(a), 0);
		}
	}

	void think(std::array<T, layerSizes[0]> const &input,
			std::array<T, outputCount> const &expectedOutput) {
		for(size_t i : util::lang::indices(activations[0])) {
			activations[0][i] = input[i];
		}

		for(size_t i : util::lang::range(1ul, layerSizes.size())) {
			for(size_t j : util::lang::indices(biases[i])) {
				T sum = biases[i][j];
				for(size_t k : util::lang::indices(weights[i][j]))
					sum += activations[i - 1][k] * weights[i][j][k];
				activations[i][j] = sigmoid(sum);
			}
		}

		cost = 0;
		for(size_t i : util::lang::indices(output)) {
			error[i] = expectedOutput[i] - output[i];
			cost += error[i] * error[i];
		}
		cost /= outputCount;

		fmt::print("[");
		for(size_t i : util::lang::indices(input)) {
			fmt::print(&" {:.2f}"[!i], input[i]);
		}
		fmt::print("] -> [");
		for(auto i : util::lang::indices(output)) {
			fmt::print(&" {:.2f}"[!i], output[i]);
		}
		fmt::print("] Expected: [");
		for(size_t i : util::lang::indices(expectedOutput)) {
			fmt::print(&" {:.2f}"[!i], expectedOutput[i]);
		}
		fmt::print("] Cost: {:.2f}\n", cost);
	}

	void train(size_t n, std::array<T, layerSizes[0]> const inputs[],
			std::array<T, outputCount> const expectedOutputs[]) {
		for(auto &a : delta->activations) {
			std::fill(begin(a), end(a), 0);
		}
		std::fill(begin(delta->error), end(delta->error), 0);

		for(size_t i : util::lang::range(0ul, n)) {
			think(inputs[i], expectedOutputs[i]);
		}

	}
};
