#include "LossFunction.h"

__device__ LossFunction::LossFunction() {}
__device__ float LossFunction::loss_func(float, int) { return 0.0f; }
__global__ void delete_loss_function(LossFunction** loss) {
	delete* loss;
}

__device__ MSELoss::MSELoss(float* d_outputs) :
	d_outputs(d_outputs)
{}
__device__ float MSELoss::loss_func(float val, int index) {
	return 2.0f * (d_outputs[index] - val);
}
__global__ void create_mse_function(LossFunction** mse_loss, float* d_outputs) {
	(*mse_loss) = new MSELoss(d_outputs);
}