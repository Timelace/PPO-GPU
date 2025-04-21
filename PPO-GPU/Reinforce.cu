#include "Reinforce.h"
#include "Reinforce.cuh"

Reinforce::Reinforce(const int network_size, int* layer_sizes, const int batch_size, const float eta, const float leaky_relu, const int max_trajectory_length, const size_t state_size, int epochs, Environment environment) :
	h_max_trajectory_length(max_trajectory_length),
	h_state_size(state_size),
	h_epochs(epochs)
{
	net = new Network(network_size, layer_sizes, batch_size, eta, leaky_relu);
	h_environment = environment;

	init_trajectories(epochs);
}

Reinforce::~Reinforce() {
	delete net;
}

void Reinforce::init_trajectories(int epochs) {
	cudaMalloc((void**)&d_trajectories_host, epochs * sizeof(TrajectorySequence));
	cudaMemcpyToSymbol(d_trajectories, &d_trajectories_host, sizeof(TrajectorySequence*), 0, cudaMemcpyHostToDevice);

	TrajectorySequence* h_trajectories = (TrajectorySequence*)malloc(epochs * sizeof(TrajectorySequence));
	for (int i = 0; i < epochs; i++) {
		h_trajectories[i] = *(new TrajectorySequence(h_max_trajectory_length, h_state_size, true));
	}
	cudaMemcpy(d_trajectories_host, h_trajectories, epochs * sizeof(TrajectorySequence), cudaMemcpyHostToDevice);
	for (int i = 0; i < epochs; i++) {
		delete &h_trajectories[i];
	}
	free(h_trajectories);
}
