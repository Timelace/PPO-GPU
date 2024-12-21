#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GraphingNetwork.h"
#include <random>
#include <ctime>

GraphingNetwork::GraphingNetwork(int network_size, int* layer_sizes, int batch_size, int epochs_per_frame, int frames, int points, float eta, float leaky_relu, std::function<float(float input)> equation) :
	m_network_size(network_size),
	m_batch_size(batch_size),
	m_epochs_per_frame(epochs_per_frame),
	m_frames(frames),
	m_points(points),
	m_eta(eta),
	m_leaky_relu(leaky_relu),
	m_equation(equation)
{
	m_layer_sizes = layer_sizes;

	net = new Network(m_network_size, m_layer_sizes, batch_size, eta, leaky_relu);
	graph = new FunctionGraphics(500, 500, -2, 1, -1, 1, m_points, m_equation);

	cudaMalloc((void**)&d_outputs_host, epochs_per_frame * batch_size * sizeof(float));

	inputs = (float*)malloc(epochs_per_frame * batch_size * sizeof(float));
	outputs = (float*)malloc(epochs_per_frame * batch_size * sizeof(float));
	graph_inputs = (float*)malloc(m_points * sizeof(float));
	graph_outputs = (float*)malloc(m_points * sizeof(float));

	cudaMalloc((void**)&mse_loss, sizeof(LossFunction**));
	create_mse_function << <1, 1 >> > (mse_loss, d_outputs_host);
	cudaDeviceSynchronize();
}

GraphingNetwork::~GraphingNetwork() {
	delete net;
	delete graph;
	cudaFree(d_outputs_host);
	free(inputs);
	free(outputs);
	free(graph_inputs);
	free(graph_outputs);
	delete_loss_function << <1, 1 >> > (mse_loss);
	cudaDeviceSynchronize();
	cudaFree(mse_loss);
}

void GraphingNetwork::start() {
	std::srand(std::time(0));
	for (int frame = 0; frame < m_frames; frame++) {
		x_lb = graph->get_x_min();
		x_ub = graph->get_x_max();
		for (int i = 0; i < m_epochs_per_frame * m_batch_size; i++) {
			float random_num = (std::rand() / (float)RAND_MAX) * (x_ub - x_lb) + x_lb;
			inputs[i] = random_num;
			outputs[i] = m_equation(random_num);
		}
		cudaMemcpy(d_outputs_host, outputs, m_epochs_per_frame * m_batch_size * sizeof(float), cudaMemcpyHostToDevice);
		net->train(m_epochs_per_frame, inputs, mse_loss);
		for (int point = 0; point < m_points; point++) {
			float x_coord = (x_ub - x_lb) / m_points * point + x_lb;
			graph_inputs[point] = x_coord;
			net->network_interface(graph_inputs + point, graph_outputs + point);
		}
		graph->set_external_points(graph_outputs);
	}
}