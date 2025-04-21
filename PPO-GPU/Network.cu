#include "Network.cuh"
#include "Network.h"

#include <climits>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <chrono>
#include <string>
#include <ctime>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#define cg cooperative_groups

#define batch_normalization 0


Network::Network(const int network_size, int* layer_sizes, const int batch_size, const float eta, const float leaky_relu, const float decay) :
	h_network_size(network_size),
	h_batch_size(batch_size),
	h_leaky_relu(leaky_relu),
	h_eta(eta)
{
	h_layer_sizes = (int*)malloc(network_size * sizeof(int));
	memcpy(h_layer_sizes, layer_sizes, network_size * sizeof(int));

	cudaMemcpyToSymbol(d_network_size, &network_size, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_batch_size, &batch_size, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_leaky_relu, &leaky_relu, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_eta, &eta, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_decay, &decay, sizeof(float), 0, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_layer_sizes_host, network_size * sizeof(int));
	cudaMemcpy(d_layer_sizes_host, layer_sizes, network_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_layer_sizes, &d_layer_sizes_host, sizeof(int*), 0, cudaMemcpyHostToDevice); //still dont understand this syntax and why we give the pointer to the destination but the memory address of the ptr for the source ??



	//allocate network parameters (idk make sure page stuff and all that is good)
	int h_largest_cumulative_layer_sum = 0;
	for (int i = 1; i < network_size; i++) {
		h_neuron_count += layer_sizes[i];
		h_weight_count += layer_sizes[i] * layer_sizes[i - 1];
		if (i > 1 && i != network_size - 1) {
			if (layer_sizes[i] + layer_sizes[i - 1] > h_largest_cumulative_layer_sum)
				h_largest_cumulative_layer_sum = layer_sizes[i] + layer_sizes[i - 1];
		}
	}

	cudaMalloc((void**)&d_neurons_host, (h_neuron_count + layer_sizes[0]) * sizeof(float));
	cudaMemset(d_neurons_host, 0, (h_neuron_count + layer_sizes[0]) * sizeof(float));
	cudaMemcpyToSymbol(d_neurons, &d_neurons_host, sizeof(float*), 0, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_biases_host, h_neuron_count * sizeof(float));
	cudaMemset(d_biases_host, 0, h_neuron_count * sizeof(float));
	cudaMemcpyToSymbol(d_biases, &d_biases_host, sizeof(float*), 0, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_gamma_host, h_neuron_count * sizeof(float));
	init_value << <1, h_neuron_count >> > (d_gamma_host, 1.0f);
	cudaMemcpyToSymbol(d_gamma, &d_gamma_host, sizeof(float*), 0, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_beta_host, h_neuron_count * sizeof(float));
	cudaMemset(d_beta_host, 0, h_neuron_count * sizeof(float));
	cudaMemcpyToSymbol(d_beta, &d_beta_host, sizeof(float*), 0, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_running_mean_host, h_neuron_count * sizeof(float));
	cudaMemset(d_running_mean_host, 0, h_neuron_count * sizeof(float));
	cudaMemcpyToSymbol(d_running_mean, &d_running_mean_host, sizeof(float*), 0, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_running_variance_host, h_neuron_count * sizeof(float));
	cudaMemset(d_running_variance_host, 0, h_neuron_count * sizeof(float));
	cudaMemcpyToSymbol(d_running_variance, &d_running_variance_host, sizeof(float*), 0, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_weights_host, h_weight_count * sizeof(float));
	int seen_weights = 0;
	for (int l = 1; l < network_size; l++) {
		int layer_weights = layer_sizes[l] * layer_sizes[l - 1];
		if (layer_weights > 1024)
			init_he << <layer_weights / 1024, 1024 >> > (d_weights_host, time(0), layer_sizes[l], seen_weights);
		seen_weights += (layer_weights / 1024) * 1024;
		init_he << <1, layer_weights - ((layer_weights / 1024) * 1024) >> > (d_weights_host, time(0), layer_sizes[l], seen_weights);
		seen_weights += layer_weights - ((layer_weights / 1024) * 1024);
	}
	//if (h_weight_count > 1024)
		//init_random << <h_weight_count / 1024, 1024 >> > (d_weights_host, time(0));
	//init_random << <1, h_weight_count - ((h_weight_count / 1024) * 1024) >> > (d_weights_host + ((h_weight_count / 1024) * 1024), time(0));
	cudaDeviceSynchronize();
	cudaMemcpyToSymbol(d_weights, &d_weights_host, sizeof(float*), 0, cudaMemcpyHostToDevice);

	int penultimate_count = h_neuron_count - h_layer_sizes[h_network_size - 1];
	cudaMemcpyToSymbol(d_neuron_pencount, &penultimate_count, sizeof(int), 0, cudaMemcpyHostToDevice);

	#ifdef _DEBUG
		// set up debugger
		#if batch_normalization == 1
		helper = new DebuggerDeviceHelper(5);
		h_debug = new Debugger(14, helper);
		#else
		helper = new DebuggerDeviceHelper(1);
		h_debug = new Debugger(6, helper);
		#endif
		cudaMemcpyToSymbol(d_helper, &helper, sizeof(DebuggerDeviceHelper*), 0, cudaMemcpyHostToDevice);
		// set up convient storages
		terminal_neuron = (bool*)malloc(h_neuron_count * sizeof(bool));
		terminal_weight = (int*)malloc(h_weight_count * sizeof(int));
		weights_connections = (int*)malloc(h_weight_count * 2 * sizeof(int));
		int ni = 0, wi = 0;
		for (int l = 1; l < h_network_size; l++) {
			for (int n = 0; n < h_layer_sizes[l]; n++) {
				for (int pn = 0; pn < h_layer_sizes[l - 1]; pn++) {
					terminal_weight[wi] = 0;
					weights_connections[wi * 2] = n;
					weights_connections[wi * 2 + 1] = pn;
					wi++;
				}
				if(wi != h_weight_count)terminal_weight[wi - 1] = 1;
				terminal_neuron[ni] = false;
				ni++;
			}
			if(wi != h_weight_count)terminal_weight[wi - 1] = 2;
			if(ni != h_neuron_count)terminal_neuron[ni - 1] = true;
		}
	#endif 

}

Network::~Network() {
	free(h_layer_sizes);
	cudaFree(d_layer_sizes_host);
	cudaFree(d_neurons_host);
	cudaFree(d_biases_host);
	cudaFree(d_gamma_host);
	cudaFree(d_beta_host);
	cudaFree(d_running_mean_host);
	cudaFree(d_running_variance_host);
	cudaFree(d_weights_host);

	#ifdef _DEBUG
		free(terminal_neuron);
		free(terminal_weight);
		free(weights_connections);
		delete h_debug;
		delete helper;
	#endif
}

void Network::network_interface(float* inputs, float* outputs) {
	// for now just assume no layer can have more then 1024 neurons cuz im lazy ~~
	cudaMemcpy(d_neurons_host, inputs, h_layer_sizes[0] * sizeof(float), cudaMemcpyHostToDevice);
	int neuron_offset = 0, p_neuron_offset = 0, weight_offset = 0;
	for (int layer = 1; layer < h_network_size; layer++) {
		feedforward_interface << <1, h_layer_sizes[layer] >> > (neuron_offset, p_neuron_offset, weight_offset, h_layer_sizes[layer], h_layer_sizes[layer - 1]);
		neuron_offset += h_layer_sizes[layer];
		p_neuron_offset += h_layer_sizes[layer - 1];
		weight_offset += h_layer_sizes[layer] * h_layer_sizes[layer - 1];
	}
	cudaDeviceSynchronize();

	cudaMemcpy(outputs, d_neurons_host + p_neuron_offset, h_layer_sizes[h_network_size - 1] * sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void feedforward_interface(int neuron_offset, int p_neuron_offset, int weight_offset, int layer_size, int p_layer_size) {
	float* neuron_ptr = d_neurons + (neuron_offset + d_layer_sizes[0] + blockDim.x * blockIdx.x + threadIdx.x);
	float* p_neuron_ptr = d_neurons + p_neuron_offset;
	float* weight_ptr = d_weights + (weight_offset + (blockDim.x * blockIdx.x + threadIdx.x) * p_layer_size);
	float bias = *(d_biases + (neuron_offset + blockDim.x * blockIdx.x + threadIdx.x));
	#if batch_normalization == 1
		float gamma = *(d_gamma + (neuron_offset + blockDim.x * blockIdx.x + threadIdx.x));
		float beta = *(d_beta + (neuron_offset + blockDim.x * blockIdx.x + threadIdx.x));
		float mean = *(d_running_mean + (neuron_offset + blockDim.x * blockIdx.x + threadIdx.x));
		float variance = *(d_running_variance + (neuron_offset + blockDim.x * blockIdx.x + threadIdx.x));
	#endif
	float neuron = 0;
	for (int pn = 0; pn < p_layer_size; pn++) {
		neuron += *(weight_ptr + pn) * *(p_neuron_ptr + pn);
	}
	neuron += bias;

	#if batch_normalzation == 1	
		neuron -= mean;
		neuron /= std::sqrt(variance + epsilon);

		neuron *= gamma;
		neuron += beta;
	#endif

	neuron = relu(neuron);

	*neuron_ptr = neuron;
}

//__device__ void network_interface_device(float* inputs, float* outputs) {
//	int neuron_offset = 0, weight_offset = 0;
//
//	for (int layer = 1; layer < d_network_size; layer++) {
//		feedforward_interface << <1, h_layer_sizes[layer] >> > (neuron_offset, p_neuron_offset, weight_offset, h_layer_sizes[layer], h_layer_sizes[layer - 1]);
//		neuron_offset += d_layer_sizes[layer];
//		p_neuron_offset += d_layer_sizes[layer - 1];
//		weight_offset += d_layer_sizes[layer] * d_layer_sizes[layer - 1];
//	}
//	cudaDeviceSynchronize();
//}

__global__ void feedforward_interface_2(float* neurons, float* p_neurons, int neuron_offset, int weight_offset, int layer_size, int p_layer_size) {
	float* neuron_ptr = d_neurons + (blockDim.x * blockIdx.x + threadIdx.x);
	float* p_neuron_ptr = d_neurons;
	float* weight_ptr = d_weights + (weight_offset + (blockDim.x * blockIdx.x + threadIdx.x) * p_layer_size);
	float bias = *(d_biases + (neuron_offset + blockDim.x * blockIdx.x + threadIdx.x));

	float neuron = 0;
	for (int pn = 0; pn < p_layer_size; pn++) {
		neuron += *(weight_ptr + pn) * *(p_neuron_ptr + pn);
	}
	neuron += bias;

	neuron = relu(neuron);

	*neuron_ptr = neuron;
}



__global__ void print_things(float* arr, int items) {
	for (int i = 0; i < items; i++) 
		printf("(%d): %f  ", i, arr[i]);
	printf("\n\n\n");
}

void Network::train(int epochs, float* inputs, LossFunction** loss) {
	size_t free = 0, total = 0;
	cudaMemGetInfo(&free, &total);

		// find max weight size per layer (needed for calculating max occupancy)
	int max_weight = 0;
	for (int i = 1; i < h_network_size; i++)
		if (h_layer_sizes[i] * h_layer_sizes[i - 1] > max_weight)
			max_weight = h_layer_sizes[i] * h_layer_sizes[i - 1];

		// max occupancy for feedforward
	int ff_blocks = 0, ff_threads = 0;
	cudaOccupancyMaxPotentialBlockSize(&ff_blocks, &ff_threads, feedforward, max_weight * sizeof(float), h_batch_size);
		// max occupancy for backprop
	int bp_blocks = 0, bp_threads = 0;
	cudaOccupancyMaxPotentialBlockSize(&bp_blocks, &bp_threads, backprop, max_weight * 2 * sizeof(float), h_batch_size);

	// threads have to be the same between all kernels because of batch size + shared memory reasons
	int threads = (ff_threads > bp_threads) ? bp_threads : ff_threads;
	// recalculate max block size (block size can differ between kernels)
	cudaOccupancyMaxPotentialBlockSize(&ff_blocks, &ff_threads, feedforward, max_weight * sizeof(float), threads);
	cudaOccupancyMaxPotentialBlockSize(&bp_blocks, &bp_threads, backprop, max_weight * 2 * sizeof(float), threads);

	// threads might now be less then batch size, so we will have to reduce batch size
	h_batch_size = threads;
	cudaMemcpyToSymbol(d_batch_size, &h_batch_size, sizeof(int), 0, cudaMemcpyHostToDevice);

	// probably not having more then 1 epoch simultanously (max i think my 1660 can do is 24 blocks with size of 1024), ill generalize this layer to allow for more epochs later

	float* neurons, * z, * z_norm, * y, * means, * variances;
	float* targets;
	cudaStream_t s1, s2, s3;
	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);
	cudaStreamCreate(&s3);

	// what goes in each stream was choosen arbitrarily based on what i think would finish fast, can look into later
	cudaMallocAsync((void**)&neurons, (h_neuron_count + h_layer_sizes[0]) * h_batch_size * sizeof(float), s1);
	cudaMemcpyAsync(neurons, inputs, h_layer_sizes[0] * h_batch_size * sizeof(float), cudaMemcpyHostToDevice, s1);
	#if batch_normalization == 1
		cudaMallocAsync((void**)&z_norm, h_neuron_count * h_batch_size * sizeof(float), s1);
	#endif

	cudaMallocAsync((void**)&z, h_neuron_count * h_batch_size * sizeof(float), s2);
	cudaMemsetAsync(z, 0, h_neuron_count * h_batch_size * sizeof(float), s2);
	#if batch_normalization == 1
		cudaMallocAsync((void**)&y, h_neuron_count * h_batch_size * sizeof(float), s2);
	#endif

	#if batch_normalization == 1
		cudaMallocAsync((void**)&means, h_neuron_count * sizeof(float), s3);
		cudaMemsetAsync(means, 0, h_neuron_count * sizeof(float), s3);
		cudaMallocAsync((void**)&variances, h_neuron_count * sizeof(float), s3);
		cudaMemsetAsync(variances, 0, h_neuron_count * sizeof(float), s3);

	#endif

	cudaMallocAsync((void**)&targets, h_layer_sizes[h_network_size - 1] * h_batch_size * sizeof(float), s3); // dont need this until later so we could probably move this after the inital paramters

	cudaStreamSynchronize(s1);
	cudaStreamSynchronize(s2);
	cudaStreamSynchronize(s3);
	cudaDeviceSynchronize();

	#ifdef _DEBUG
		h_debug->add_parameter("weights", "Debug-Files/Network-Parameters/weights.txt", h_weight_count);
		h_debug->add_parameter("biases", "Debug-Files/Network-Parameters/biases.txt", h_neuron_count);
		h_debug->add_parameter("neurons", "Debug-Files/Feedforward-Variables/neurons.txt", (h_neuron_count + h_layer_sizes[0]) * h_batch_size);
		h_debug->add_parameter("zs", "Debug-Files/Feedforward-Variables/zs.txt", h_neuron_count * h_batch_size);
		h_debug->add_parameter("d_neurons", "Debug-Files/Backpropagation-Variables/d_neurons.txt", (h_neuron_count + h_layer_sizes[0]) * h_batch_size);
		h_debug->add_device_parameter("d_zs", "Debug-Files/Backpropagation-Variables/d_zs.txt", h_neuron_count * h_batch_size);

		h_debug->print_file_parameter("biases", d_biases_host, [&terminal_neuron = this->terminal_neuron](int index, float value) -> std::string {
			std::string s = "";
			if (index == 0)
				s += "\nStarting new Epoch\n";
			s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
			if (terminal_neuron[index])
				s += "\n=====================================================================\n";
			return s;
			});

		h_debug->print_file_parameter("weights", d_weights_host, [&terminal_weight = this->terminal_weight, &weights_connections = this->weights_connections](int index, float value) -> std::string {
			std::string s = "";
			if (index == 0)
				s += "\nStarting new Epoch\n";
			s += "(" + std::to_string(weights_connections[index * 2]) + "-" + std::to_string(weights_connections[index * 2 + 1]) + "): " + std::to_string(value) + " | ";
			if (terminal_weight[index] == 1) {
				s += "\n---------------------------------------------------------------------\n";
			}
			else if (terminal_weight[index] == 2) {
				s += "\n=====================================================================\n";
			}
			return s;
		});
		#if batch_normalization == 1
		h_debug->add_parameter("z_norms", "Debug-Files/Feedforward-Variables/z_norms.txt", h_neuron_count * h_batch_size);
		h_debug->add_parameter("ys", "Debug-Files/Feedforward-Variables/ys.txt", h_neuron_count * h_batch_size);
		h_debug->add_parameter("means", "Debug-Files/Feedforward-Variables/means.txt", h_neuron_count);
		h_debug->add_parameter("variances", "Debug-Files/Feedforward-Variables/variances.txt", h_neuron_count);
		h_debug->add_device_parameter("d_z_norms", "Debug-Files/Backpropagation-Variables/d_z_norms.txt", h_neuron_count * h_batch_size);
		h_debug->add_device_parameter("d_ys", "Debug-Files/Backpropagation-Variables/d_ys.txt", h_neuron_count * h_batch_size);
		h_debug->add_device_parameter("d_means", "Debug-Files/Backpropagation-Variables/d_means.txt", h_neuron_count);
		h_debug->add_device_parameter("d_variances", "Debug-Files/Backpropagation-Variables/d_variances.txt", h_neuron_count);
		#endif
	#endif

		int epoch_offset = 0;
	for (int epoch = 1; epoch <= epochs; epoch++) {
		if (epoch % 100 == 0) printf("done with %d\n", epoch);

		// feedforward
		int neuron_offset = 0, p_neuron_offset = 0, weight_offset = 0;
		for (int layer = 1; layer < h_network_size - 1; layer++) {
			int neurons_remaining = h_layer_sizes[layer];
			while (neurons_remaining != 0) {
				int n = (ff_blocks > neurons_remaining) ? neurons_remaining : ff_blocks;
				#if batch_normalization == 1
					feedforward << <n, threads, h_layer_sizes[layer] * h_layer_sizes[layer - 1] * sizeof(float) >> > (neurons, z, z_norm, y, means, variances,
					neuron_offset, p_neuron_offset, weight_offset, h_layer_sizes[layer], h_layer_sizes[layer - 1]);
				#else
					feedforward << <n, threads, h_layer_sizes[layer] * h_layer_sizes[layer - 1] * sizeof(float) >> > (neurons, z, nullptr, nullptr, nullptr, nullptr,
					neuron_offset, p_neuron_offset, weight_offset, h_layer_sizes[layer], h_layer_sizes[layer - 1]);
				#endif
				neuron_offset += n;
				weight_offset += n * h_layer_sizes[layer - 1];
				neurons_remaining -= n;
			}
			p_neuron_offset += h_layer_sizes[layer - 1];
		}

		int neurons_remaining = h_layer_sizes[h_network_size - 1];
		while (neurons_remaining != 0) {
			int n = (ff_blocks > neurons_remaining) ? neurons_remaining : ff_blocks;
			#if batch_normalization == 1
			feedforward_output << <n, threads, h_layer_sizes[h_network_size - 1] * h_layer_sizes[h_network_size - 1 - 1] * sizeof(float) >> > (neurons, z, z_norm, y, means, variances,
				neuron_offset, p_neuron_offset, weight_offset, h_layer_sizes[h_network_size - 1], h_layer_sizes[h_network_size - 1 - 1], epoch_offset, loss);
			#else
			feedforward_output << <n, threads, h_layer_sizes[h_network_size - 1] * h_layer_sizes[h_network_size - 1 - 1] * sizeof(float) >> > (neurons, z, nullptr, nullptr, nullptr, nullptr,
				neuron_offset, p_neuron_offset, weight_offset, h_layer_sizes[h_network_size - 1], h_layer_sizes[h_network_size - 1 - 1], epoch_offset, loss);
			#endif
			neuron_offset += n;
			weight_offset += n * h_layer_sizes[h_network_size - 1 - 1];
			neurons_remaining -= n;
		}
		p_neuron_offset += h_layer_sizes[h_network_size - 1 - 1];

		#ifdef _DEBUG
		if (epoch % debug_print_epoch == 0) {
			cudaDeviceSynchronize();
			h_debug->print_file_parameter_custom_index("neurons", neurons,
				[&batch_size = this->h_batch_size, &l0 = this->h_layer_sizes[0], &neuron_count = this->h_neuron_count](int index)->int {
					if (index == ((l0 + neuron_count) * batch_size - 1)) // check for terminal index as otherwise it would wrap back to 0
						return index;
					return (index * batch_size) % ((l0 + neuron_count) * batch_size - 1);
				},
				[&batch_size = this->h_batch_size, &l0 = this->h_layer_sizes[0], &terminal_neuron = this->terminal_neuron](int index, float value)->std::string {
					std::string s = "";
					if (index == 0)
						s += "Starting new Epoch";
					if (index / batch_size == 0)
						s += "\n\n=====================================================================\nBatch " + std::to_string(index) + ":\n\n";
					s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
					if (terminal_neuron[index / batch_size - l0] || (index / batch_size) == l0 - 1)
						s += "\n=====================================================================\n";
					return s;
				});

			h_debug->print_file_parameter_custom_index("zs", z,
				[&batch_size = this->h_batch_size, &neuron_count = this->h_neuron_count](int index)->int {
					if (index == (neuron_count * batch_size - 1)) // check for terminal index as otherwise it would wrap back to 0
						return index;
					return (index * batch_size) % ((neuron_count)*batch_size - 1);
				},
				[&batch_size = this->h_batch_size, &terminal_neuron = this->terminal_neuron](int index, float value)->std::string {
					std::string s = "";
					if (index == 0)
						s += "Starting new Epoch";
					if (index / batch_size == 0)
						s += "\n\n=====================================================================\nBatch " + std::to_string(index) + ":\n\n";
					s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
					if (terminal_neuron[index / batch_size])
						s += "\n=====================================================================\n";
					return s;
				});
#if batch_normalization == 1
			h_debug->print_file_parameter_custom_index("z_norms", z_norm,
				[&batch_size = this->h_batch_size, &neuron_count = this->h_neuron_count](int index)->int {
					if (index == (neuron_count * batch_size - 1)) // check for terminal index as otherwise it would wrap back to 0
						return index;
					return (index * batch_size) % ((neuron_count)*batch_size - 1);
				},
				[&batch_size = this->h_batch_size, &terminal_neuron = this->terminal_neuron](int index, float value)->std::string {
					std::string s = "";
					if (index == 0)
						s += "Starting new Epoch";
					if (index / batch_size == 0)
						s += "\n\n=====================================================================\nBatch " + std::to_string(index) + ":\n\n";
					s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
					if (terminal_neuron[index / batch_size])
						s += "\n=====================================================================\n";
					return s;
				});
			h_debug->print_file_parameter_custom_index("ys", y,
				[&batch_size = this->h_batch_size, &neuron_count = this->h_neuron_count](int index)->int {
					if (index == (neuron_count * batch_size - 1)) // check for terminal index as otherwise it would wrap back to 0
						return index;
					return (index * batch_size) % ((neuron_count)*batch_size - 1);
				},
				[&batch_size = this->h_batch_size, &terminal_neuron = this->terminal_neuron](int index, float value)->std::string {
					std::string s = "";
					if (index == 0)
						s += "Starting new Epoch";
					if (index / batch_size == 0)
						s += "\n\n=====================================================================\nBatch " + std::to_string(index) + ":\n\n";
					s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
					if (terminal_neuron[index / batch_size])
						s += "\n=====================================================================\n";
					return s;
				});
			h_debug->print_file_parameter("means", means, [&terminal_neuron = this->terminal_neuron](int index, float value) -> std::string {
				std::string s = "";
				if (index == 0)
					s += "Starting new Epoch\n";
				s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
				if (terminal_neuron[index])
					s += "\n=====================================================================\n";
				return s;
				});
			h_debug->print_file_parameter("variances", variances, [&terminal_neuron = this->terminal_neuron](int index, float value) -> std::string {
				std::string s = "";
				if (index == 0)
					s += "Starting new Epoch\n";
				s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
				if (terminal_neuron[index])
					s += "\n=====================================================================\n";
				return s;
				});
#endif
			}
		#endif

		/*	// convient for backpropagation for neuron offset + weight offset to not include the final layer
		neuron_offset -= h_layer_sizes[h_network_size - 1];
		weight_offset -= h_layer_sizes[h_network_size - 1] * h_layer_sizes[h_network_size - 2];

		// calculate output errors
		int neurons_remaining = h_layer_sizes[h_network_size - 1];
		int output_offset = 0;
		while (neurons_remaining != 0) {
			int n = (mse_blocks > neurons_remaining) ? neurons_remaining : mse_blocks;
			mse_derivative << <n, threads >> > (neurons, targets, neuron_offset, output_offset);
			neuron_offset += n;
			output_offset += n;
			neurons_remaining -= n;
		}*/


			// better for backprop if we dont include final layer in neuron offset
		neuron_offset -= h_layer_sizes[h_network_size - 1];
		p_neuron_offset -= h_layer_sizes[h_network_size - 2];
		weight_offset -= h_layer_sizes[h_network_size - 1] * h_layer_sizes[h_network_size - 2];

		// backprop
		for (int layer = h_network_size - 1; layer > 0; layer--) {
			int neurons_remaining = h_layer_sizes[layer];
			while (neurons_remaining != 0) {
				int n = (bp_blocks > neurons_remaining) ? neurons_remaining : bp_blocks;
				#if batch_normalization == 1
					backprop << <n, threads, h_layer_sizes[layer] * h_layer_sizes[layer - 1] * 2 * sizeof(float) >> > (
					neurons, z, z_norm, y, means, variances, neuron_offset, p_neuron_offset, weight_offset, h_layer_sizes[layer], h_layer_sizes[layer - 1]);
				#else
				float* def_real_ptr_trust = nullptr;
				void* kernel_args[] = { &neurons, &z, &def_real_ptr_trust, &def_real_ptr_trust, &def_real_ptr_trust, &def_real_ptr_trust, &neuron_offset, &p_neuron_offset, &weight_offset, &h_layer_sizes[layer], &h_layer_sizes[layer - 1] };
				cudaLaunchCooperativeKernel((void*)backprop, n, threads, kernel_args, h_layer_sizes[layer] * h_layer_sizes[layer - 1] * 2 * sizeof(float));
					//backprop << <n, threads, h_layer_sizes[layer] * h_layer_sizes[layer - 1] * 2 * sizeof(float) >> > (
					//neurons, z, nullptr, nullptr, nullptr, nullptr, neuron_offset, p_neuron_offset, weight_offset, h_layer_sizes[layer], h_layer_sizes[layer - 1]);
				#endif
				neuron_offset += n;
				weight_offset += n * h_layer_sizes[layer - 1];
				neurons_remaining -= n;
			}

			neuron_offset -= (h_layer_sizes[layer] + h_layer_sizes[layer - 1]);
			if (layer != 1) {
				p_neuron_offset -= h_layer_sizes[layer - 2];
				weight_offset -= (h_layer_sizes[layer] * h_layer_sizes[layer - 1] + h_layer_sizes[layer - 1] * h_layer_sizes[layer - 2]);
			}
				
		}

		#ifdef _DEBUG
		if (epoch % debug_print_epoch == 0) {
			cudaDeviceSynchronize();
			h_debug->print_file_parameter_custom_index("d_neurons", neurons,
				[&batch_size = this->h_batch_size, &l0 = this->h_layer_sizes[0], &neuron_count = this->h_neuron_count](int index)->int {
					if (index == ((l0 + neuron_count) * batch_size - 1)) // check for terminal index as otherwise it would wrap back to 0
						return index;
					return (index * batch_size) % ((l0 + neuron_count) * batch_size - 1);
				},
				[&batch_size = this->h_batch_size, &l0 = this->h_layer_sizes[0], &terminal_neuron = this->terminal_neuron](int index, float value)->std::string {
					std::string s = "";
					if (index == 0)
						s += "Starting new Epoch";
					if (index / batch_size == 0)
						s += "\n\n=====================================================================\nBatch " + std::to_string(index) + ":\n\n";
					s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
					if (terminal_neuron[index / batch_size - l0] || (index / batch_size) == l0 - 1)
						s += "\n=====================================================================\n";
					return s;
				});
			h_debug->print_file_parameter_custom_index("d_zs", nullptr,
				[&batch_size = this->h_batch_size, &neuron_count = this->h_neuron_count](int index)->int {
					if (index == (neuron_count * batch_size - 1)) // check for terminal index as otherwise it would wrap back to 0
						return index;
					return (index * batch_size) % ((neuron_count)*batch_size - 1);
				},
				[&batch_size = this->h_batch_size, &terminal_neuron = this->terminal_neuron](int index, float value)->std::string {
					std::string s = "";
					if (index == 0)
						s += "Starting new Epoch";
					if (index / batch_size == 0)
						s += "\n\n=====================================================================\nBatch " + std::to_string(index) + ":\n\n";
					s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
					if (terminal_neuron[index / batch_size])
						s += "\n=====================================================================\n";
					return s;
				});
#if batch_normalization == 1
			h_debug->print_file_parameter_custom_index("d_z_norms", nullptr,
				[&batch_size = this->h_batch_size, &neuron_count = this->h_neuron_count](int index)->int {
					if (index == (neuron_count * batch_size - 1)) // check for terminal index as otherwise it would wrap back to 0
						return index;
					return (index * batch_size) % ((neuron_count)*batch_size - 1);
				},
				[&batch_size = this->h_batch_size, &terminal_neuron = this->terminal_neuron](int index, float value)->std::string {
					std::string s = "";
					if (index == 0)
						s += "Starting new Epoch";
					if (index / batch_size == 0)
						s += "\n\n=====================================================================\nBatch " + std::to_string(index) + ":\n\n";
					s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
					if (terminal_neuron[index / batch_size])
						s += "\n=====================================================================\n";
					return s;
				});
			h_debug->print_file_parameter_custom_index("d_ys", nullptr,
				[&batch_size = this->h_batch_size, &neuron_count = this->h_neuron_count](int index)->int {
					if (index == (neuron_count * batch_size - 1)) // check for terminal index as otherwise it would wrap back to 0
						return index;
					return (index * batch_size) % ((neuron_count)*batch_size - 1);
				},
				[&batch_size = this->h_batch_size, &terminal_neuron = this->terminal_neuron](int index, float value)->std::string {
					std::string s = "";
					if (index == 0)
						s += "Starting new Epoch";
					if (index / batch_size == 0)
						s += "\n\n=====================================================================\nBatch " + std::to_string(index) + ":\n\n";
					s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
					if (terminal_neuron[index / batch_size])
						s += "\n=====================================================================\n";
					return s;
				});
			h_debug->print_file_parameter("d_means", nullptr, [&terminal_neuron = this->terminal_neuron](int index, float value) -> std::string {
				std::string s = "";
				if (index == 0)
					s += "Starting new Epoch\n";
				s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
				if (terminal_neuron[index])
					s += "\n=====================================================================\n";
				return s;
				});
			h_debug->print_file_parameter("d_variances", nullptr, [&terminal_neuron = this->terminal_neuron](int index, float value) -> std::string {
				std::string s = "";
				if (index == 0)
					s += "Starting new Epoch\n";
				s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
				if (terminal_neuron[index])
					s += "\n=====================================================================\n";
				return s;
				});
#endif
			h_debug->print_file_parameter("biases", d_biases_host, [&terminal_neuron = this->terminal_neuron](int index, float value) -> std::string {
				std::string s = "";
				if (index == 0)
					s += "\nStarting new Epoch\n";
				s += "(" + std::to_string(index) + "): " + std::to_string(value) + " | ";
				if (terminal_neuron[index])
					s += "\n=====================================================================\n";
				return s;
				});

			h_debug->print_file_parameter("weights", d_weights_host, [&terminal_weight = this->terminal_weight, &weights_connections = this->weights_connections](int index, float value) -> std::string {
				std::string s = "";
				if (index == 0)
					s += "\nStarting new Epoch\n";
				s += "(" + std::to_string(weights_connections[index * 2]) + "-" + std::to_string(weights_connections[index * 2 + 1]) + "): " + std::to_string(value) + " | ";
				if (terminal_weight[index] == 1) {
					s += "\n---------------------------------------------------------------------\n";
				}
				else if (terminal_weight[index] == 2) {
					s += "\n=====================================================================\n";
				}
				return s;
				});
		}
		#endif


		cudaDeviceSynchronize();

		// set up next epoch

		if (epoch != epochs) {
			cudaMemcpy(neurons, inputs + (h_layer_sizes[0] * h_batch_size) * (epoch), h_layer_sizes[0] * h_batch_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemset(z, 0, h_neuron_count * h_batch_size * sizeof(float));
			#if batch_normalization == 1
				cudaMemset(means, 0, h_neuron_count * sizeof(float));
				cudaMemset(variances, 0, h_neuron_count * sizeof(float));
			#endif
			epoch_offset += h_layer_sizes[h_network_size - 1] * threads;
		}

	}

	// free memory
	cudaFree(neurons);
	cudaFree(z);
	#if batch_normalization == 1
		cudaFree(z_norm);
		cudaFree(y);
		cudaFree(means);
		cudaFree(variances);
	#endif
	cudaFree(targets);

	cudaStreamDestroy(s1);
	cudaStreamDestroy(s2);
	cudaStreamDestroy(s3);

}

// performs loss function on output neurons
__global__ void feedforward(float* neurons, float* zs, float* z_norms, float* ys, float* means, float* variances, int neuron_offset, int p_neuron_offset, int weight_offset, int layer_size, int p_layer_size) {
	extern __shared__ float parameters[];
	auto group = cg::this_thread_block();
	cg::memcpy_async(group, parameters, d_weights + weight_offset, (layer_size * p_layer_size * sizeof(float)));
		
	int threads = blockDim.x;
	float* neuron = neurons + ((neuron_offset + d_layer_sizes[0] + blockIdx.x) * threads + threadIdx.x);
	float* p_neuron = neurons + (p_neuron_offset * threads + threadIdx.x);

	float* z = zs + ((neuron_offset + blockIdx.x) * threads + threadIdx.x);
	float l_z = 0;

	#if batch_normalization == 1
		float* z_norm = z_norms + ((neuron_offset + blockIdx.x) * threads + threadIdx.x);
		float l_z_norm;
		float* y = ys + ((neuron_offset + blockIdx.x) * threads + threadIdx.x);
		float l_y;

		__shared__ float mean;
		__shared__ float variance;

		__shared__ float gamma;
		__shared__ float beta;
	#endif

	__shared__ float bias;
	
	if (threadIdx.x == 0) {
		#if batch_normalization == 1
			mean = 0;
			variance = 0;

			gamma = *(d_gamma + neuron_offset + blockIdx.x);
			beta = *(d_beta + neuron_offset + blockIdx.x);
		#endif
		bias = *(d_biases + neuron_offset + blockIdx.x);
	}


	cg::wait(group);

	for (int pn = 0; pn < p_layer_size; pn++) {
		l_z += *(p_neuron + (blockDim.x * pn)) * (parameters[blockIdx.x * p_layer_size + pn]); // p_neuron horrible for caching btw
	}
	__syncthreads(); // ensures shared variables are properly assigned (cg::wait(group) might already take care of this)
	l_z += bias; // add bias


	#if batch_normalization == 1
		// mean + variance
		atomicAdd(&mean, l_z / blockDim.x);
		__syncthreads();
		atomicAdd(&variance, (l_z - mean) * (l_z - mean) / (blockDim.x - 1));
		__syncthreads();

		l_z_norm = (l_z - mean) / (std::sqrt(variance + epsilon)); // normalize

		l_y = l_z_norm * gamma + beta; // gamma beta

		*neuron = relu(l_y); // activation

		*z_norm = l_z_norm;
		*y = l_y;
	#else
		*neuron = relu(l_z); // activation
	#endif
	*z = l_z;

	#if batch_normalization == 1
		// update running mean + variance
		// update global mean + variance
		int index = 0;
		__shared__ int temp;
		if (threadIdx.x == 0) temp = 1;
		__syncthreads();
		index = atomicAdd(&temp, 1);
		if (index == blockDim.x) { // last thread to run
			*(means + neuron_offset + blockIdx.x) = mean;
			*(variances + neuron_offset + blockIdx.x) = variance;
			*(d_running_mean + neuron_offset + blockIdx.x) = *(d_running_mean + neuron_offset + blockIdx.x) * d_decay + (mean) * (1 - d_decay);
			*(d_running_variance + neuron_offset + blockIdx.x) = *(d_running_variance + neuron_offset + blockIdx.x) * d_decay + (variance) * (1 - d_decay);
		}
	#endif
}

__global__ void feedforward_output(float* neurons, float* zs, float* z_norms, float* ys, float* means, float* variances, int neuron_offset, int p_neuron_offset, int weight_offset, int layer_size, int p_layer_size, int epoch_offset, LossFunction** __restrict__ loss) {
	extern __shared__ float parameters[];
	auto group = cg::this_thread_block();
	cg::memcpy_async(group, parameters, d_weights + weight_offset, (layer_size * p_layer_size * sizeof(float)));

	int threads = blockDim.x;
	float* neuron = neurons + ((neuron_offset + d_layer_sizes[0] + blockIdx.x) * threads + threadIdx.x);
	float* p_neuron = neurons + (p_neuron_offset * threads + threadIdx.x);

	float* z = zs + ((neuron_offset + blockIdx.x) * threads + threadIdx.x);
	float l_z = 0;

	#if batch_normalization == 1
	float* z_norm = z_norms + ((neuron_offset + blockIdx.x) * threads + threadIdx.x);
	float l_z_norm;
	float* y = ys + ((neuron_offset + blockIdx.x) * threads + threadIdx.x);
	float l_y;

	__shared__ float mean;
	__shared__ float variance;

	__shared__ float gamma;
	__shared__ float beta;
	#endif

	__shared__ float bias;

	if (threadIdx.x == 0) {
	#if batch_normalization == 1
		mean = 0;
		variance = 0;

		gamma = *(d_gamma + neuron_offset + blockIdx.x);
		beta = *(d_beta + neuron_offset + blockIdx.x);
	#endif
		bias = *(d_biases + neuron_offset + blockIdx.x);
	}


	cg::wait(group);

	for (int pn = 0; pn < p_layer_size; pn++) {
		l_z += *(p_neuron + (blockDim.x * pn)) * (parameters[blockIdx.x * p_layer_size + pn]); // p_neuron horrible for caching btw
	}
	__syncthreads(); // ensures shared variables are properly assigned (cg::wait(group) might already take care of this)
	l_z += bias; // add bias

	#if batch_normalization == 1
	// mean + variance
	atomicAdd(&mean, l_z / blockDim.x);
	__syncthreads();
	atomicAdd(&variance, (l_z - mean) * (l_z - mean) / (blockDim.x - 1));
	__syncthreads();

	l_z_norm = (l_z - mean) / (std::sqrt(variance + epsilon)); // normalize

	l_y = l_z_norm * gamma + beta; // gamma beta

	*neuron = relu(l_y); // activation

	*z_norm = l_z_norm;
	*y = l_y;
	#else
	* z = l_z;
	l_z = relu(l_z); // activation
	#endif
	* neuron = (*loss)->loss_func(l_z, (neuron_offset + blockIdx.x - d_neuron_pencount) * threads + threadIdx.x + epoch_offset);

	#if batch_normalization == 1
	// update running mean + variance
	// update global mean + variance
	int index = 0;
	__shared__ int temp;
	if (threadIdx.x == 0) temp = 1;
	__syncthreads();
	index = atomicAdd(&temp, 1);
	if (index == blockDim.x) { // last thread to run
		*(means + neuron_offset + blockIdx.x) = mean;
		*(variances + neuron_offset + blockIdx.x) = variance;
		*(d_running_mean + neuron_offset + blockIdx.x) = *(d_running_mean + neuron_offset + blockIdx.x) * d_decay + (mean) * (1 - d_decay);
		*(d_running_variance + neuron_offset + blockIdx.x) = *(d_running_variance + neuron_offset + blockIdx.x) * d_decay + (variance) * (1 - d_decay);
	}
	#endif


}

// storing the neuron errors in the neuron array
// also updates network paramters directly
__global__ void backprop(float* neurons, float* zs, float* z_norms, float* ys, float* means, float* variances, int neuron_offset, int p_neuron_offset, int weight_offset, int layer_size, int p_layer_size) {
	int threads = blockDim.x;
	float* neuron = neurons + ((neuron_offset + d_layer_sizes[0] + blockIdx.x) * threads + threadIdx.x);
	float* p_neuron = neurons + (p_neuron_offset * threads + threadIdx.x);
	float* z = zs + ((neuron_offset + blockIdx.x) * threads + threadIdx.x);
	#if batch_normalization == 1
		float* z_norm = z_norms + ((neuron_offset + blockIdx.x) * threads + threadIdx.x);
		float* y = ys + ((neuron_offset + blockIdx.x) * threads + threadIdx.x);
	#endif

	extern __shared__ float parameters[]; //weights + weight_grad temp storage
	auto group = cg::this_thread_block();
	auto grid = cg::this_grid();

	__shared__ float* weights;
	#if batch_normalization == 1
		__shared__ float mean;
		__shared__ float variance;
		__shared__ float gamma;
	#endif

	// fast access local storage for gradients (will be fully updated at the end)
		// note note - going to go with this for now, an alternative strategy would be to allow for each thread to have its own place in the shared array and then at the very end have 1 thread loop through the entire shared array and do 1 atomic add at the end ? 
		// or split it up into smaller groups, like every 100 threads will coallece its own group and atomic add that, will see when optimizing
		// for now this should be faster though still cuz its atomic adds to shared mem as oppose to global mem
	__shared__ float* weights_grad;
	__shared__ float bias_grad;
	#if batch_normalization == 1
		__shared__ float gamma_grad;
		__shared__ float beta_grad;
	#endif

	cg::memcpy_async(group, parameters, d_weights + weight_offset, layer_size * p_layer_size * sizeof(float));
	if (threadIdx.x == 0) {
		weights = parameters;
		weights_grad = parameters + (layer_size * p_layer_size);
		#if batch_normalization == 1
			mean = *(means + neuron_offset + blockIdx.x);
			variance = *(variances + neuron_offset + blockIdx.x);
			gamma = *(d_gamma + neuron_offset + blockIdx.x);

			gamma_grad = 0;
			beta_grad = 0;
		#endif
		bias_grad = 0;
	}

	__syncthreads();

	float error = 0;

	#if batch_normalization == 1
		error = (*neuron) * relu_derivative(*y); // activation derivative

		#ifdef _DEBUG
			d_helper->set_device_param("d_ys", ((neuron_offset + blockIdx.x) * threads + threadIdx.x), error);
		#endif

		// gamma + beta gradient
		atomicAdd(&gamma_grad, error * (*z_norm)); // mmmmmmm not happy about this mmmmm
		atomicAdd(&beta_grad, error);

		error *= gamma; // z_norm derivative

		#ifdef _DEBUG
			d_helper->set_device_param("d_z_norms", ((neuron_offset + blockIdx.x) * threads + threadIdx.x), error);
		#endif

		// big arithmetical calculation caching
		float z_minus_mean = *z - mean;

		float var_error; // variance derivative
		__shared__ float z_minus_mean_times_dx_norm_sum; // lmao
		if (threadIdx.x == 0) z_minus_mean_times_dx_norm_sum = 0;
		__syncthreads();
		atomicAdd(&z_minus_mean_times_dx_norm_sum, error * z_minus_mean); // tearing my hair out
		__syncthreads();
		var_error = z_minus_mean_times_dx_norm_sum * -0.5 * (1.0f / (std::sqrt(variance + epsilon) * variance));

		float mean_error; // mean derivative
		__shared__ float dz_norm_sum;
		__shared__ float z_minus_mean_sum;
		if (threadIdx.x == 0) {
			dz_norm_sum = 0;
			z_minus_mean_sum = 0;
		}
		__syncthreads();
		atomicAdd(&dz_norm_sum, error);
		atomicAdd(&z_minus_mean_sum, z_minus_mean); // i have no words left to say
		__syncthreads();
		mean_error = dz_norm_sum * (-1.0f / std::sqrt(variance + epsilon)) + var_error * -2.0f * z_minus_mean_sum * (1.0f / blockDim.x);
	
		// z derivative
			// i could probably actually calculate the rough clock cycles it takes to do this to see if caching would be faster
		error = error * (1.0f / std::sqrt(variance + epsilon)) + var_error * 2 * z_minus_mean * (1.0f / blockDim.x) + mean_error * (1.0f / blockDim.x);
	#else
		error = (*neuron) * relu_derivative(*z); // activation derivative
	#endif

	#ifdef _DEBUG
		d_helper->set_device_param("d_zs", ((neuron_offset + blockIdx.x) * threads + threadIdx.x), error);
	#endif

	// weights + bias derivative
	for (int pn = 0; pn < p_layer_size; pn++) {
		atomicAdd(weights_grad + pn, *(p_neuron + blockDim.x * pn) * error); // weights derivative
		//*(p_neuron + blockDim.x * pn) = *(weights + pn) * error; // prior neurons derivative
	}
	atomicAdd(&bias_grad, error); // bias derivative

	cg::wait(group);

	// for testing purposes
	grid.sync();
	for (int pn = 0; pn < p_layer_size; pn++) {
		*(p_neuron + blockDim.x * pn) = 0;
	}
	grid.sync();
	for (int pn = 0; pn < p_layer_size; pn++) {
		atomicAdd((p_neuron + blockDim.x * pn), *(weights + pn + blockIdx.x * p_layer_size) * error); // prior neurons derivative
	}

	*neuron = error;

	// update global network parameters - should switch to a reduction
	int index = 0;
	__shared__ int temp;
	if (threadIdx.x == 0) temp = 1;
	__syncthreads();
	index = atomicAdd(&temp, 1);
	if (index == blockDim.x) { // last thread to run
		for (int pn = 0; pn < p_layer_size; pn++) {
			atomicAdd(d_weights + weight_offset + (blockIdx.x * p_layer_size) + pn, *(weights_grad + pn) * -d_eta / d_batch_size);
		}
		atomicAdd(d_biases + neuron_offset + blockIdx.x, bias_grad * -d_eta / d_batch_size);
		#if batch_normalization == 1
			atomicAdd(d_gamma + neuron_offset + blockIdx.x, gamma_grad * -d_eta / d_batch_size);
			atomicAdd(d_beta + neuron_offset + blockIdx.x, beta_grad * -d_eta / d_batch_size);

			#ifdef _DEBUG
				d_helper->set_device_param("d_means", neuron_offset + blockIdx.x, mean_error);
				d_helper->set_device_param("d_variances", neuron_offset + blockIdx.x, var_error);
			#endif
		#endif
	}

}

__device__ float relu(float val) {
	return (val > 0) ? val : val * d_leaky_relu;
}

__device__ float relu_derivative(float val) {
	return (val > 0) ? 1 : d_leaky_relu;
}

__global__ void init_random(float* float_arr, unsigned long long seed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float* rand = float_arr + id;
	curandState state;
	curand_init(seed, id, 0, &state);
	*rand = (curand_uniform(&state) * 2.0f) - 1.0f;
}

__global__ void init_he(float* float_arr, unsigned long long seed, int layer_size, int array_offset) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float* rand = float_arr + id + array_offset;
	curandState state;
	curand_init(seed, id, 0, &state);
	*rand = (curand_normal(&state) * (2.0f / layer_size));
}

__global__ void init_value(float* float_arr, float val) {
	float_arr[blockIdx.x * blockDim.x + threadIdx.x] = val;
}

void Network::set_biases(float* h_biases) {
	cudaMemcpy(d_biases_host, h_biases, h_neuron_count * sizeof(float), cudaMemcpyHostToDevice);
}

void Network::set_weights(float* h_weights) {
	cudaMemcpy(d_weights_host, h_weights, h_weight_count * sizeof(float), cudaMemcpyHostToDevice);
}
