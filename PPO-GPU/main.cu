#include <cmath>
#include <functional>
#include "FunctionGraphics.h"
#include "GraphingNetwork.h"
#include "BasicFullyConnectedTester.h"
#include "LossFunction.h"
#include "Network.h"
#include <iostream>
#include <random>
#include <ctime>
#include "SnakeGame.h"
#include "CarGame.h"


float equation(float input) {
	return std::exp(input) * std::sin(1 - input);
}

//__device__ float* d_outputs;
//
//__device__ float loss_func(float val, int index) {
//	return 2.0f * (d_outputs[index] - val);
//}


int main() {
	CarGame game(900, 900, 10, 10);

	//int network_size = 4;
	//int layer_sizes[] = {1,24,24,1};
	//int batch_size = 1023;
	//int epochs_per_frame = 1000;
	//int frames = 50;
	//int points = 150;
	//float eta = -0.001;
	//float leaky_relu = 0.01;

	//SnakeGame snake(500, 500, 50, 50, 0.05);

	//FunctionGraphics graph(500, 500, -1, 1, -1, 1, 150, equation);

	//BasicFullyConnectedTester network_tester;
	//network_tester.create_from_scratch(network_size, layer_sizes, batch_size, eta, leaky_relu, epochs_per_frame);
	//network_tester.run_network();

	//GraphingNetwork graph_net(network_size, layer_sizes, batch_size, epochs_per_frame, frames, points, eta, leaky_relu, equation);
	//graph_net.start();

	/*Network net(network_size, layer_sizes, batch_size, eta, leaky_relu);

	float* inputs = (float*)malloc(epochs_per_frame * batch_size * sizeof(float));
	float* outputs = (float*)malloc(epochs_per_frame * batch_size * sizeof(float));
	float* d_outputs_host;
	cudaMalloc((void**)&d_outputs_host, epochs_per_frame * batch_size * sizeof(float));
	cudaMemcpyToSymbol(d_outputs, &d_outputs_host, sizeof(float*), 0, cudaMemcpyHostToDevice);
	std::srand(std::time(0));
	for (int i = 0; i < epochs_per_frame * batch_size; i++) {
		inputs[i] = (rand() / (float)RAND_MAX);
		outputs[i] = equation(inputs[i]);
	}
	cudaMemcpy(d_outputs_host, outputs, epochs_per_frame * batch_size * sizeof(float), cudaMemcpyHostToDevice);
	net.train(epochs_per_frame, inputs);
	free(inputs);
	free(outputs);
	inputs = (float*)malloc(sizeof(float));
	outputs = (float*)malloc(sizeof(float));
	for (int i = 0; i < 100; i++) {
		*inputs = (rand() / (float)RAND_MAX);
		net.network_interface(inputs, outputs);
		printf("%f\t%f\n", *outputs, equation(*inputs));
	}*/
	std::cin.get();


	return 0;
}