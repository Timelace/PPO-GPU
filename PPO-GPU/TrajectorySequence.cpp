#include "TrajectorySequence.h"

#include <cstdlib> 
#include <cstring> // since memcpy is part of cstring for some reason

__host__ __device__ TrajectorySequence::TrajectorySequence(int max_length, size_t state_size, bool record_policy):
	state_size(state_size),
	record_policy(record_policy)
{
	state_seq = (float*)malloc(max_length * state_size * sizeof(float));
	reward_seq = (float*)malloc(max_length * sizeof(float));
	if (record_policy)
		policy_seq = (float*)malloc(max_length * sizeof(float));
}

__host__ __device__ TrajectorySequence::~TrajectorySequence() {
	free(state_seq);
	free(reward_seq);
	if (record_policy)
		free(policy_seq);
}

__host__ __device__ void TrajectorySequence::add_trajectory(float* s, float reward, float policy) {
	memcpy(state_seq + (c_index * state_size), s, state_size * sizeof(float));
	reward_seq[c_index] = reward;
	if (record_policy)
		policy_seq[c_index] = policy;
	c_index++;
}

__host__ __device__ int TrajectorySequence::get_size() {
	return c_index;
}

__host__ __device__ void TrajectorySequence::calculate_qvals(float* q_out, float lambda) {
	float total_val = 0;
	for (int i = c_index - 1; i >= 0; i--) {
		total_val = total_val * lambda + reward_seq[i];
		q_out[i] = total_val;
	}
}

__host__ __device__ float* TrajectorySequence::get_state(int index) {
	return state_seq + (index * state_size);
}

__host__ __device__ float* TrajectorySequence::get_all_states() {
	return state_seq;
}