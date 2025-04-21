#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include "DebuggerDeviceHelper.cuh"
#include "LossFunction.h"

// variables
__device__ __constant__ int d_network_size;
__device__ int* d_layer_sizes; // in global mem since constant memory needs to be known at compile time (maybe it could work w/ some preprocessor stuff but I digress [would be fast though in const memory]) anyways could also try moving to texture mem
int* d_layer_sizes_host; // the cringe memory cuz you cant malloc gmem on the host :(
__device__ __constant__ int d_batch_size;
__device__ __constant__ float d_leaky_relu;
__device__ __constant__ float d_eta;
__device__ __constant__ float epsilon = 0.0001;
__device__ __constant__ float d_decay;
__device__ __constant__ int d_largest_cumulative_layer_sizes;

// network parameters
__device__ float* d_neurons;
__device__ float* d_weights;
__device__ float* d_biases;
__device__ float* d_gamma;
__device__ float* d_beta;
__device__ float* d_running_mean;
__device__ float* d_running_variance;

	// network parameters on host
float* d_neurons_host;
float* d_weights_host;
float* d_biases_host;
float* d_gamma_host;
float* d_beta_host;
float* d_running_mean_host;
float* d_running_variance_host;

#ifdef _DEBUG
	__device__ DebuggerDeviceHelper* d_helper;
#endif

__device__ int d_neuron_pencount;

// functions
__global__ void init_random(float* float_arr, unsigned long long seed);
__global__ void init_value(float* arr, float val);

__global__ void feedforward(float* neurons, float* zs, float* z_norms, float* ys, float* means, float* variances, int neuron_offset, int p_neuron_offset, int weight_offset, int layer_size, int p_layer_size);
__global__ void feedforward_output(float* neurons, float* zs, float* z_norms, float* ys, float* means, float* variances, int neuron_offset, int p_neuron_offset, int weight_offset, int layer_size, int p_layer_size, int epoch_offset, LossFunction** __restrict__ loss);
__global__ void backprop(float* neurons, float* zs, float* z_norms, float* ys, float* means, float* variances, int neuron_offset, int p_neuron_offset, int weight_offset, int layer_size, int p_layer_size);

__device__ float relu(float val);
__device__ float relu_derivative(float val);

__global__ void feedforward_interface(int neuron_offset, int p_neuron_offset, int weight_offset, int layer_size, int p_layer_size);
__global__ void feedforward_interface_2(float* neurons, float* p_neurons, int neuron_offset, int weight_offset, int layer_size, int p_layer_size);

__global__ void init_he(float* float_arr, unsigned long long seed, int layer_size, int array_offset);