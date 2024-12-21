#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <functional>

#include "DebuggerDeviceHelper.cuh"

class Debugger {
public:

	Debugger();
	Debugger(int params_num);
	Debugger(int params_nums, DebuggerDeviceHelper* helper);
	~Debugger();

	/**
	*  Sets up a custom debugging parameter. Allocates host memory as storage for printing.
	* 
	*  @param name - Unique identifier for the parameter
	*  @param file_path - File path to file where debugging information will be printed
	*  @param elems - Total number of elements in the parameter
	*/
	void add_parameter(std::string name, std::string file_path, int elems);
	/**
	*  Sets up a custom debugging parameter. Does not allocate host memory for storage, instead assumes the user has allocated storage for the parameter on the host.
	*
	*  @param name - Unique identifier for the parameter
	*  @param file_path - File path to file where debugging information will be printed
	*  @param parameter_ptr - Pointer to already allocated host memory to store parameter
	*  @param elems - Total number of elements in the parameter
	*/
	void add_parameter(std::string name, std::string file_path, float* parameter_ptr, int elems);

	void add_device_parameter(std::string name, std::string file_path, int elems);

	/**
	*  Outputs data to a specific file. Copies the device allocated host pointer to host memory. Loops through entire parameter in sequential 0-indexed order. Only prints the value given by the formatter for every index.
	* 
	*  @param name - Unique identifier for the parameter
	*  @param d_host_ptr - Device allocated host pointer
	*  @param formatter - Custom formatter. Takes in the current index of the parameter and the value at that index.
	*/
	void print_file_parameter(std::string name, float* d_host_ptr, std::function<std::string(int index, float value)> formatter);
	/**
	*  Outputs data to a specific file. Copies the device allocated host pointer to host memory. Loops through entire parameter by order determined by indexer. Indexer is provided with every index in parameter in sequential 0-indexed order, the output of the indexer is used that the index provided for the formatter. Only prints the value given by the formatter for every index.
	*
	*  @param name - Unique identifier for the parameter
	*  @param d_host_ptr - Device allocated host pointer
	*  @param indexer - Custom indexer. Takes in the current index and outputs a custom index used in the formatter
	*  @param formatter - Custom formatter. Takes in the current index of the parameter and the value at that index.
	*/
	void print_file_parameter_custom_index(std::string name, float* d_host_ptr, std::function<int(int)> indexer, std::function<std::string(int index, float value)> formatter);
	

private:

	const int m_params_num = 5;

	std::unordered_map<std::string, int> index_map;
	int c_index = 0;

	std::ofstream** streams;
	float** data;
	int* elements;

	std::vector<int> allocated_ptrs;

	DebuggerDeviceHelper* helper = nullptr;

};