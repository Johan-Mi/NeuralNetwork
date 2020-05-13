#include <ctime>

#include "net.hpp"

int main() {
	srand(std::time(0));

	NeuralNet<float> net(2, 1);

	constexpr float inputs[] = {
		0.f, 0.f,
		0.f, 1.f,
		1.f, 0.f,
		1.f, 1.f,
	};

	constexpr float outputs[] = {
		0.f,
		1.f,
		1.f,
		0.f,
	};

	for(size_t i = 0; i < 3; i++)
		net.train(4, inputs, outputs);

	return 0;
}
