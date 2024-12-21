#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>

//__device__ extern int* d_c_index;

class DebuggerDeviceHelper {
public:
	void* operator new(size_t len);
	void operator delete(void* ptr);

	DebuggerDeviceHelper(int device_params);

	void add_device_param(std::string name, int elems);

	float* get_param(std::string);

	bool contains(std::string name);

 	__device__ void set_device_param(const char* name, int index, float value);

	char** get_index_map();
	char** index_map;

private:

	int c_index = 0;
	float** data;

	int get_index_host(const char* name);
	__device__ int get_index_device(const char* name);
};