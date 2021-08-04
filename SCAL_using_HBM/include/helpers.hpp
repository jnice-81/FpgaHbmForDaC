#pragma once

#include "include.hpp"

/*
Loads the device
*/
void loadu280fpga(cl::Device &device, cl::Context &context) {
	std::vector<cl::Device> xildevice = xcl::get_xil_devices();
	bool success = false;
	for(size_t i = 0; i < xildevice.size(); i++) {
		if(xildevice[i].getInfo<CL_DEVICE_NAME>() == "xilinx_u280_xdma_201920_3") {
			 device = xildevice[i];
			 success = true;
			 break;
		}
	}
	if(!success) {		
		std::cerr << "Failed to load device" << std::endl;
		std::exit(-1);
	}
	cl_int error;
	OCL_CHECK(error, context = cl::Context(device, NULL, NULL, NULL, &error));
}

/*
Write the .xclbin to the device
*/
void applyProgram(cl::Device &device, cl::Context &context, cl::Program &program, std::string program_path) {
	auto buf = xcl::read_binary_file(program_path);
	cl_int err;
	OCL_CHECK(err, program = cl::Program(context, {device}, {{buf.data(), buf.size()}}, NULL, &err));
}

/*
returns the smallest n (natural number) such that n >= currentvalue and n % divisor == 0
*/
size_t roundUpToMultiplesOf(const size_t divisor, const size_t currentvalue) {
	size_t newsize = currentvalue;
	if(newsize % divisor != 0) {
		newsize = (newsize + divisor) - (newsize % divisor);
	}
	return newsize;
}
