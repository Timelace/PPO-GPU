#pragma once

#include "Network.h"
#include <functional>
#include "FunctionGraphics.h"
#include "LossFunction.h"

class GraphingNetwork {
public:
	GraphingNetwork(int network_size, int* layer_sizes, int batch_size, int epochs_per_frame, int frames, int points, float eta, float leaky_relu, std::function<float(float input)> equation);
	~GraphingNetwork();
	void start();
private:
	const int m_network_size;
	int* m_layer_sizes;
	const int m_batch_size;
	const int m_epochs_per_frame;
	const int m_frames;
	const int m_points;
	const float m_eta;
	const float m_leaky_relu;

	float* inputs;
	float* outputs;
	float* graph_inputs;
	float* graph_outputs;

	float x_lb = -2;
	float x_ub = 1;

	Network* net;
	FunctionGraphics* graph;

	float* d_outputs_host;

	std::function<float(float input)> m_equation;

	LossFunction** mse_loss;
};
