#pragma once
#include "Network.h"
#include "LossFunction.h"
#include <string>

class BasicFullyConnectedTester {
public:
	BasicFullyConnectedTester();
	~BasicFullyConnectedTester();

	void create_from_scratch(int network_size, int* layer_sizes, int batch_size, float eta, float leaky_relu, int epochs);
	void create_from_file(const std::string& structure_file_name, const std::string& biases_file_name, const std::string& weights_file_name);

	void run_network();
	void test_network(int test_samples);

private:
	Network* net = nullptr;

	float* inputs = nullptr;
	float* outputs = nullptr;
	float* d_outputs_host = nullptr;

	int epochs;
	int batch_size;

	LossFunction** mse_loss;
};