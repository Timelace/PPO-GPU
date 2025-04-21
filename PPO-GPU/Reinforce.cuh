#include "Network.h"
#include "Environment.h"
#include "TrajectorySequence.h"

__device__ TrajectorySequence* d_trajectories;
TrajectorySequence* d_trajectories_host;