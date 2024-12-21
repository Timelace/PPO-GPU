#include "BasicFullyConnectedTester.h"
#include <fstream>
#include <random>
#include <chrono>
#include <iostream>

BasicFullyConnectedTester::BasicFullyConnectedTester() {
	cudaMalloc((void**)&mse_loss, sizeof(LossFunction**));
	create_mse_function << <1, 1 >> > (mse_loss, d_outputs_host);
	cudaDeviceSynchronize();
}

BasicFullyConnectedTester::~BasicFullyConnectedTester() {
	if (net != nullptr)
		delete net;
	if (inputs != nullptr)
		free(inputs);
	if (outputs != nullptr)
		free(outputs);
	if (d_outputs_host != nullptr)
		cudaFree(d_outputs_host);
	delete_loss_function << <1, 1 >> > (mse_loss);
	cudaDeviceSynchronize();
	cudaFree(mse_loss);
}

void BasicFullyConnectedTester::create_from_scratch(int network_size, int* layer_sizes, int batch_size, float eta, float leaky_relu, int epochs) {
	net = new Network(network_size, layer_sizes, batch_size, eta, leaky_relu);
	this->epochs = epochs;
	this->batch_size = batch_size;
	inputs = (float*)malloc(epochs * batch_size * sizeof(float));
	outputs = (float*)malloc(epochs * batch_size * sizeof(float));
	cudaMalloc((void**)&d_outputs_host, epochs * batch_size * sizeof(float));
}

void BasicFullyConnectedTester::create_from_file(const std::string& structure_file_name, const std::string& biases_file_name, const std::string& weights_file_name) {
	std::ifstream struct_file(structure_file_name);
	std::string str;
	std::getline(struct_file, str);
	int network_size = std::stoi(str);
	int* layer_sizes = (int*)malloc(network_size * sizeof(int));
	std::getline(struct_file, str);
	for (int i = 0; i < network_size; i++) {
		layer_sizes[i] = std::stoi(str.substr(0, str.find(",")));
		str.erase(0, str.find(","));
	}
	std::getline(struct_file, str);
	int batch_size = std::stoi(str);
	this->batch_size = batch_size;
	std::getline(struct_file, str);
	float eta = std::stof(str);
	std::getline(struct_file, str);
	float leaky_relu = std::stof(str);
	net = new Network(network_size, layer_sizes, batch_size, eta, leaky_relu);
	std::getline(struct_file, str);
	epochs = std::stoi(str);
	inputs = (float*)malloc(epochs * batch_size * sizeof(float));
	outputs = (float*)malloc(epochs * batch_size * sizeof(float));
	cudaMalloc((void**)&d_outputs_host, epochs * batch_size * sizeof(float));

	int neuron_count = 0, weights_count = 0;
	for (int l = 1; l < network_size; l++) {
		neuron_count += layer_sizes[l];
		weights_count += layer_sizes[l] * layer_sizes[l - 1];
	}
	std::ifstream biases_file(biases_file_name);
	float* biases = (float*)malloc(neuron_count * sizeof(float));
	for (int i = 0; i < neuron_count; i++) {
		std::getline(biases_file, str, '|');
		biases[i] = std::stof(str);
	}
	net->set_biases(biases);
	std::ifstream weights_file(weights_file_name);
	float* weights = (float*)malloc(weights_count * sizeof(float));
	for (int i = 0; i < weights_count; i++) {
		std::getline(weights_file, str, '|');
		weights[i] = std::stof(str);
	}
	net->set_weights(weights);

	struct_file.close();
	free(layer_sizes);
	free(biases);
	free(weights);
}

void BasicFullyConnectedTester::run_network() {
	std::srand(std::time(0));
	for (int i = 0; i < batch_size * epochs; i++) {
		float random_num = (std::rand() / (float)RAND_MAX);
		inputs[i] = random_num;
		//outputs[i] = equation(random_num);
	}
	cudaMemcpy(d_outputs_host, outputs, epochs * batch_size * sizeof(float), cudaMemcpyHostToDevice);
	net->train(epochs, inputs, mse_loss);
}

void BasicFullyConnectedTester::test_network(int test_samples) {
	float* test_inputs = (float*)malloc(sizeof(float));
	float* test_outputs = (float*)malloc(sizeof(float));
	for (int i = 0; i < test_samples; i++) {
		float random_num = (std::rand() / (float)RAND_MAX);
		*test_inputs = random_num;
		net->network_interface(test_inputs, test_outputs);
		//printf("Predicted: %f\tActual: %f\n", *test_outputs, equation(*test_inputs));
	}
	free(test_inputs);
	free(test_outputs);
}