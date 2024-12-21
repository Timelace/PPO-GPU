#pragma once

#include "Network.h"
#include "Environment.h"

class Reinforce {
public:
	Reinforce(const int network_size, int* layer_sizes, const int batch_size, const float eta, const float leaky_relu, const int max_trajectory_length, const size_t state_size, Environment environment);
	~Reinforce();

	void gather_trajectories(int epochs);
private:

	const int h_max_trajectory_length;
	const size_t h_state_size;

	Network* net;
	Environment h_environment;

};
