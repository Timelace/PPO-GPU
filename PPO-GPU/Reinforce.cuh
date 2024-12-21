#include "Network.h"
#include "Environment.h"
#include "TrajectorySequence.h"

__device__ TrajectorySequence* d_trajectories;
TrajectorySequence* d_trajectories_host;

__global__ void init_trajectories(int index_offset, int max_length, size_t state_size);