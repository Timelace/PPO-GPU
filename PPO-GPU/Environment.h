#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Environment {
public:
	__host__ __device__ float* get_state();
	__host__ __device__ float* get_action_space();
	__host__ __device__ void set_action(float action);
	__host__ __device__ void next_game_state();
};
