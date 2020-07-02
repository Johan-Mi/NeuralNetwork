#pragma once

#include <memory>
#include <vector>
#include <experimental/array>
#include <range/v3/view.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/algorithm/fill.hpp>
#include <fmt/format.h>

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
		for(auto[a, b, l] : ranges::views::zip(
					activations, biases, layerSizes)) {
			a.resize(l);
			b.resize(l);
		}
	}

	NeuralNet()
	: delta(std::make_unique<NeuralNet<T, LayerSizes...>>(0)) {
		for(auto[a, b, w, l] : ranges::views::zip(
					activations, biases, weights, layerSizes)) {
			a.resize(l);
			b.resize(l);
			w.resize(l);
		}
		for(size_t const i : ranges::views::iota(1ul, layerSizes.size())) {
			for(auto &a : weights[i]) {
				a.resize(layerSizes[i - 1]);
			}
		}

		for(auto &a : weights) {
			for(auto &b : a) {
				ranges::generate(b, randNormalized<T>);
			}
		}

		for(auto &a : biases) {
			ranges::fill(a, 0);
		}
	}

	void think(std::array<T, layerSizes[0]> const &input,
			std::array<T, outputCount> const &expectedOutput) {
		ranges::copy(begin(input), end(input), begin(activations[0]));

		for(size_t const i : ranges::views::iota(1ul, layerSizes.size())) {
			for(size_t const j : ranges::views::iota(0ul, biases[i].size())) {
				T const sum = ranges::accumulate(ranges::views::zip(
							activations[i - 1], weights[i][j])
						| ranges::views::transform([](auto i){
							auto[a, w] = i;
							return a * w;
							}), biases[i][j]);
				activations[i][j] = sigmoid(sum);
			}
		}

		cost = 0;
		for(auto[e, x, o] : ranges::views::zip(
					error, expectedOutput, output)) {
			e = x - o;
			cost += e * e;
		}
		cost /= outputCount;

		if constexpr(true) {
			fmt::print("[");
			for(auto &a : input) {
				if(&a == &input[0]) {
					fmt::print("{:.2f}", a);
				} else {
					fmt::print(" {:.2f}", a);
				}
			}
			fmt::print("] -> [");
			for(auto &a : output) {
				if(&a == &output[0]) {
					fmt::print("{:.2f}", a);
				} else {
					fmt::print(" {:.2f}", a);
				}
			}
			fmt::print("] Expected: [");
			for(auto &a : expectedOutput) {
				if(&a == &expectedOutput[0]) {
					fmt::print("{:.2f}", a);
				} else {
					fmt::print(" {:.2f}", a);
				}
			}
			fmt::print("] Cost: {:.3f}\n", cost);
		}
	}

	void train(size_t const n, std::array<T, layerSizes[0]> const inputs[],
			std::array<T, outputCount> const expectedOutputs[]) {
		for(auto &a : delta->activations) {
			ranges::fill(a, 0);
		}
		ranges::fill(delta->error, 0);

		for(size_t const i : ranges::views::iota(0ul, n)) {
			think(inputs[i], expectedOutputs[i]);
		}

	}
};
