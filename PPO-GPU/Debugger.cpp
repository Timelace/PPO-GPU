#include "Debugger.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

Debugger::Debugger() {}

Debugger::Debugger(int params_num) :
	m_params_num(params_num) 
{
	streams = (std::ofstream**)malloc(params_num * sizeof(std::ofstream*));
	data = (float**)malloc(params_num * sizeof(float*));
	elements = (int*)malloc(params_num * sizeof(int));
}

Debugger::Debugger(int params_num, DebuggerDeviceHelper* helper) :
	m_params_num(params_num),
	helper(helper)
{
	streams = (std::ofstream**)malloc(params_num * sizeof(std::ofstream*));
	data = (float**)malloc(params_num * sizeof(float*));
	elements = (int*)malloc(params_num * sizeof(int));
}

Debugger::~Debugger() {
	/*for (int i = 0; i < m_params_num; i++) {
		streams[i]->close();
		delete streams[i];
	}
	for (int i : allocated_ptrs) {
		free(data[i]);
	}
	free(elements);
	free(data);
	free(streams);*/
}

void Debugger::add_parameter(std::string name, std::string file_path, int elems) {
	index_map[name] = c_index;
	streams[c_index] = new std::ofstream(file_path);
	data[c_index] = (float*)malloc(elems * sizeof(float));
	elements[c_index] = elems;
	allocated_ptrs.push_back(c_index);
	c_index++;
}

void Debugger::add_parameter(std::string name, std::string file_path, float* parameter_ptr, int elems) {
	index_map[name] = c_index;
	streams[c_index] = new std::ofstream(file_path);
	data[c_index] = parameter_ptr;
	elements[c_index] = elems;
	c_index++;
}

void Debugger::add_device_parameter(std::string name, std::string file_path, int elems) {
	if (helper == nullptr)
		return;
	index_map[name] = c_index;
	streams[c_index] = new std::ofstream(file_path);
	elements[c_index] = elems;
	helper->add_device_param(name, elems);
	c_index++;
}

void Debugger::print_file_parameter(std::string name, float* d_host_ptr, std::function<std::string(int index, float value)> formatter) {
	int index = index_map[name];
	if (helper != nullptr && helper->contains(name)) {
		data[index] = helper->get_param(name);
	}
	else if (d_host_ptr != nullptr) {
		cudaMemcpy(data[index], d_host_ptr, elements[index] * sizeof(float), cudaMemcpyDeviceToHost);
	}
	for (int i = 0; i < elements[index]; i++) {
		*(streams[index]) << formatter(i, data[index][i]);
	}
}

void Debugger::print_file_parameter_custom_index(std::string name, float* d_host_ptr, std::function<int(int)> indexer, std::function<std::string(int index, float value)> formatter) {
	cudaDeviceSynchronize();
	int index = index_map[name];
	if (helper != nullptr && helper->contains(name)) {
		data[index] = helper->get_param(name);
	}
	else if (d_host_ptr != nullptr) {
		cudaMemcpy(data[index], d_host_ptr, elements[index] * sizeof(float), cudaMemcpyDeviceToHost);
	}
	for (int i = 0; i < elements[index]; i++) {
		int indexed_i = indexer(i);
		*(streams[index]) << formatter(indexed_i, data[index][indexed_i]);
	}
	cudaDeviceSynchronize();
}