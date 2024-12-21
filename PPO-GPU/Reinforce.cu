#include "Reinforce.h"
#include "Reinforce.cuh"

Reinforce::Reinforce(const int network_size, int* layer_sizes, const int batch_size, const float eta, const float leaky_relu, const int max_trajectory_length, const size_t state_size, Environment environment) :
	h_max_trajectory_length(max_trajectory_length),
	h_state_size(state_size)
{
	net = new Network(network_size, layer_sizes, batch_size, eta, leaky_relu);
	h_environment = environment;

}

Reinforce::~Reinforce() {
	delete net;
}

void Reinforce::gather_trajectories(int epochs) {
	cudaMalloc((void**)&d_trajectories_host, epochs * sizeof(TrajectorySequence));
	cudaMemcpyToSymbol(d_trajectories, &d_trajectories_host, sizeof(TrajectorySequence*), 0, cudaMemcpyHostToDevice);

}

__global__ void init_trajectories(int index_offset, int max_length, size_t state_size) {
	d_trajectories[index_offset + blockIdx.x * blockDim.x + threadIdx.x] = *(new TrajectorySequence(max_length, state_size, true));
}
