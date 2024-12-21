#pragma once

class TrajectorySequence {
public:
	TrajectorySequence(int max_length, size_t state_size, bool record_policy = true);
	~TrajectorySequence();

	void add_trajectory(float* s, float reward, float policy = 0.0f);
	int get_size();
	void calculate_qvals(float* q_out, float lambda);
	float* get_state(int index);
	float* get_all_states();

private:

	int c_index = 0;

	bool record_policy;
	size_t state_size;

	float* state_seq;
	float* reward_seq;
	float* policy_seq;
};