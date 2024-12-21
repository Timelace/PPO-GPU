#pragma once

#include "Debugger.h"
#include "LossFunction.h"

class Network {
protected:
public:
	Network(const int network_size, int* layer_sizes, const int batch_size, const float eta, const float leaky_relu, const float decay = 0);
	~Network();

	void train(int epochs, float* inputs, LossFunction** loss);
	void network_interface(float* inputs, float* outputs);

	void set_weights(float* h_weights);
	void set_biases(float* h_biases);

	int debug_print_epoch = 1;


private:

	int h_neuron_count = 0, h_weight_count = 0;

	const int h_network_size;
	int h_batch_size;
	int* h_layer_sizes;
	const float h_leaky_relu, h_eta;

	#ifdef _DEBUG
		Debugger* h_debug;	
		DebuggerDeviceHelper* helper;

		bool* terminal_neuron;
		int* terminal_weight;
		int* weights_connections;
	#endif

};

