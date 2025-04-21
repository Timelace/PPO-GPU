#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class TrajectorySequence {
public:
	__host__ __device__ TrajectorySequence(int max_length, size_t state_size, bool record_policy = true);
	__host__ __device__ ~TrajectorySequence();

	__host__ __device__ void add_trajectory(float* s, float reward, float policy = 0.0f);
	__host__ __device__ int get_size();
	__host__ __device__ void calculate_qvals(float* q_out, float lambda);
	__host__ __device__ float* get_state(int index);
	__host__ __device__ float* get_all_states();

private:

	int c_index = 0;

	bool record_policy;
	size_t state_size;

	float* state_seq;
	float* reward_seq;
	float* policy_seq;
};