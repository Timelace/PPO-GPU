#include "DebuggerDeviceHelper.cuh"

#include <cstring>

__device__ int d_c_index = 0;

void* DebuggerDeviceHelper::operator new(size_t len) {
	void* ptr;
	cudaMallocManaged(&ptr, len);
	cudaDeviceSynchronize();
	return ptr;
}

void DebuggerDeviceHelper::operator delete(void* ptr) {
	cudaDeviceSynchronize();
	cudaFree(ptr);
}

DebuggerDeviceHelper::DebuggerDeviceHelper(int device_params) {
	cudaMallocManaged(&index_map, device_params * sizeof(char*));
	cudaMallocManaged((void**)&data, device_params * sizeof(float*));
}

void DebuggerDeviceHelper::add_device_param(std::string name, int elems) {
	cudaMallocManaged(&index_map[c_index], (name.length() + 1) * sizeof(char));
	memcpy(index_map[c_index], name.c_str(), (name.length() + 1) * sizeof(char));
	cudaMallocManaged((void**)&data[c_index], elems * sizeof(float));
	c_index++;
	cudaMemcpyToSymbol(d_c_index, &c_index, sizeof(int), 0, cudaMemcpyHostToDevice);
}

char** DebuggerDeviceHelper::get_index_map() {
	return index_map;
}

float* DebuggerDeviceHelper::get_param(std::string name) {
	int index = get_index_host(name.c_str());
	if (index == -1)
		return nullptr;
	return data[index];
}

bool DebuggerDeviceHelper::contains(std::string name) {
	return get_index_host(name.c_str()) != -1;
}

int DebuggerDeviceHelper::get_index_host(const char* name) {
	for (int i = 0; i < c_index; i++) {
		int c_char = 0;
		while (name[c_char] != '\0' && index_map[i][c_char] != '\0') {
			if (name[c_char] != index_map[i][c_char])
				goto end;
			c_char++;
		}
		return i;
	end:;
	}
	return -1;
}

__device__ void DebuggerDeviceHelper::set_device_param(const char* name, int index, float value) {
	int data_index = get_index_device(name);
	if (data_index == -1)
		return;
	data[data_index][index] = value;
}

__device__ int DebuggerDeviceHelper::get_index_device(const char* name) {
	for (int i = 0; i < d_c_index; i++) {
		int c_char = 0;
		while (name[c_char] != '\0' && index_map[i][c_char] != '\0') {
			if (name[c_char] != index_map[i][c_char])
				goto end;
			c_char++;
		}
		return i;
	end:;
	}
	return -1;
}