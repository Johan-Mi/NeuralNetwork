#pragma once

#include <memory>
#include <vector>
#include <experimental/array>
#include <fmt/format.h>

#include "range.hpp"
#include "misc.hpp"

template<class T, size_t... LayerSizes>
struct NeuralNet {
	constexpr static auto layerSizes
		= std::experimental::make_array(LayerSizes...);
	constexpr static size_t outputCount = layerSizes[layerSizes.size() - 1];
	T cost = 0;
	std::array<std::vector<T>, layerSizes.size()> activations;
	std::array<std::vector<T>, layerSizes.size()> biases;
	std::array<std::vector<std::vector<T>>, layerSizes.size()> weights;
	std::vector<T> const &output = activations[layerSizes.size() - 1];
	std::array<T, outputCount> error;
	std::unique_ptr<NeuralNet> delta;

	constexpr NeuralNet(char const) {
		// Used to create the delta network
		for(size_t const i : util::lang::indices(layerSizes)) {
			activations[i].resize(layerSizes[i]);
			biases[i].resize(layerSizes[i]);
		}
	}

	NeuralNet()
	: delta(std::make_unique<NeuralNet<T, LayerSizes...>>(0)) {
		for(size_t const i : util::lang::indices(layerSizes)) {
			activations[i].resize(layerSizes[i]);
			biases[i].resize(layerSizes[i]);
			weights[i].resize(layerSizes[i]);
		}
		for(size_t const i : util::lang::range(1ul, layerSizes.size())) {
			for(auto &a : weights[i]) {
				a.resize(layerSizes[i - 1]);
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
		std::copy(begin(input), end(input), begin(activations[0]));

		for(size_t const i : util::lang::range(1ul, layerSizes.size())) {
			for(size_t const j : util::lang::indices(biases[i])) {
				T sum = biases[i][j];
				for(size_t const k : util::lang::indices(weights[i][j])) {
					sum += activations[i - 1][k] * weights[i][j][k];
				}
				activations[i][j] = sigmoid(sum);
			}
		}

		cost = 0;
		for(size_t const i : util::lang::indices(output)) {
			error[i] = expectedOutput[i] - output[i];
			cost += error[i] * error[i];
		}
		cost /= outputCount;

		if constexpr(true) {
			fmt::print("[");
			for(auto &a : input) {
				if(&a == &input[0]) [[unlikely]] {
					fmt::print("{:.2f}", a);
				} else [[likely]] {
					fmt::print(" {:.2f}", a);
				}
			}
			fmt::print("] -> [");
			for(auto &a : output) {
				if(&a == &output[0]) [[unlikely]] {
					fmt::print("{:.2f}", a);
				} else [[likely]] {
					fmt::print(" {:.2f}", a);
				}
			}
			fmt::print("] Expected: [");
			for(auto &a : expectedOutput) {
				if(&a == &expectedOutput[0]) [[unlikely]] {
					fmt::print("{:.2f}", a);
				} else [[likely]] {
					fmt::print(" {:.2f}", a);
				}
			}
			fmt::print("] Cost: {:.3f}\n", cost);
		}
	}

	void train(size_t const n, std::array<T, layerSizes[0]> const inputs[],
			std::array<T, outputCount> const expectedOutputs[]) {
		for(auto &a : delta->activations) {
			std::fill(begin(a), end(a), 0);
		}
		std::fill(begin(delta->error), end(delta->error), 0);

		for(size_t const i : util::lang::range(0ul, n)) {
			think(inputs[i], expectedOutputs[i]);
		}

	}
};
