#include <ctime>

#include "net.hpp"

int main() {
	srand(std::time(0));

	NeuralNet<float, 2, 3, 3, 1> net;

	constexpr std::array<float, 2> inputs[] = {
		0.f, 0.f,
		0.f, 1.f,
		1.f, 0.f,
		1.f, 1.f,
	};

	constexpr std::array<float, 1> outputs[] = {
		0.f,
		1.f,
		1.f,
		0.f,
	};

	for(size_t i = 0; i < 10; i++)
		net.train(4, inputs, outputs);

	return 0;
}
