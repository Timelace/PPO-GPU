#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class LossFunction {
protected:
	__device__ LossFunction();
public:
	__device__ float virtual loss_func(float, int);
};
__global__ void delete_loss_function(LossFunction** loss);

class MSELoss : public LossFunction {
public:
	__device__ MSELoss(float* d_outputs);

	__device__ float loss_func(float val, int index) override;
private:
	float* d_outputs;
};

__global__ void create_mse_function(LossFunction** mse_loss, float* d_outputs);

