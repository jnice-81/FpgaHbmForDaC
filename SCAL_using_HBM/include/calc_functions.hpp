#pragma once

#include "include.hpp"

void calc_cpu(const std::vector<vecType, aligned_allocator<vecType>> &vec_1, std::vector<vecType, aligned_allocator<vecType>> &out_1, const vecType scale) {
	const vecType *vec = vec_1.data();
	vecType *out = out_1.data();
	const size_t vsize = vec_1.size();
	for(size_t i = 0; i < vsize; i++) {
		out[i] = scale * vec[i];
	}
}

double calc_fpga(std::vector<vecType, aligned_allocator<vecType>> &vec,std::vector<vecType, aligned_allocator<vecType>> &out, 
	vecType scale, cl::Context &context, cl::Program &program, cl::Device &device) {
	//Executes scal on the fpga. Expects vec.size() % 16 = 0, and out.size() == vec.size()

	cl_int err;

	assert(vec.size() % 16 == 0);
	assert(vec.size() == out.size());

	OCL_CHECK(err, cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
	OCL_CHECK(err, cl::Kernel kernel = cl::Kernel(program, "scal", &err));

	OCL_CHECK(err, cl::Buffer vec_in(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
			sizeof(vecType)*vec.size(), vec.data(), &err));
	OCL_CHECK(err, cl::Buffer vec_out(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
		sizeof(vecType)*out.size(), out.data(), &err));

	unsigned int xsize = (unsigned int)vec.size();
	OCL_CHECK(err, err= kernel.setArg(0, vec_in));
	OCL_CHECK(err, err= kernel.setArg(1, vec_out));
	OCL_CHECK(err, err= kernel.setArg(2, xsize));
	OCL_CHECK(err, err= kernel.setArg(3, scale));

	auto start = std::chrono::high_resolution_clock::now();

	OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({vec_in}, 0));
	OCL_CHECK(err, err = queue.enqueueTask(kernel));
	OCL_CHECK(err, err = queue.finish());

	auto end = std::chrono::high_resolution_clock::now();
	auto timeneeded = std::chrono::duration<double>(end - start);

	OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({vec_out}, CL_MIGRATE_MEM_OBJECT_HOST));
	OCL_CHECK(err, err = queue.finish());

	return timeneeded.count();
}

double execKernels(std::vector<cl::Kernel> &kernels, cl::CommandQueue queue) {
	size_t kernelcount = kernels.size();
	cl_int err;

	auto start = std::chrono::high_resolution_clock::now();
	for(size_t i = 0; i < kernelcount; i++) {
		OCL_CHECK(err, err = queue.enqueueTask(kernels[i]));
	}
	OCL_CHECK(err, err = queue.finish());
	auto end = std::chrono::high_resolution_clock::now();

	auto timeneeded = std::chrono::duration<double>(end - start);
	return timeneeded.count();
}

double calc_fpgahbm16kernel(std::vector<vecType, aligned_allocator<vecType>> &vec,std::vector<vecType, aligned_allocator<vecType>> &out,
vecType scale, cl::Context &context, cl::Program &program, cl::Device &device) {
	cl_int err;
	const size_t comp_dev = 16;

	assert(vec.size() % (16 * 1024) == 0);

	MemorySplitHBMAdapter<vecType> inAdapter([](unsigned int i) {return 2*i;}, comp_dev, 
		vec, context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY);
	MemorySplitHBMAdapter<vecType> outAdapter([](unsigned int i) {return 2*i+1;}, comp_dev, 
		out, context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_WRITE_ONLY);

	size_t kernelcount =  inAdapter.getInstantiatedBlocks();
	std::vector<cl::Kernel> kernels(kernelcount);

	#ifdef DEBUG
		OCL_CHECK(err, cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
	#else
		OCL_CHECK(err, cl::CommandQueue queue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
	#endif

	inAdapter.copyToDeviceAsync(queue);
	for(size_t i = 0; i < kernelcount; i++) {
		std::string kernel_name = std::string("scal") + ":{" + "scal_" + std::to_string(i+1) + "}";
		OCL_CHECK(err, kernels[i] = cl::Kernel(program, kernel_name.c_str(), &err));

		unsigned long long xsize = inAdapter.getSplitBlock(i).size();
		OCL_CHECK(err, err= kernels[i].setArg(0, inAdapter.getBuffer(i)));
		OCL_CHECK(err, err= kernels[i].setArg(1, outAdapter.getBuffer(i)));
		OCL_CHECK(err, err= kernels[i].setArg(2, xsize));
		OCL_CHECK(err, err= kernels[i].setArg(3, scale));

		#ifdef DEBUGVERBOSE
		std::cout << "Kernel " << i << " ready" << std::endl;
		#endif

	}
	OCL_CHECK(err, err = queue.finish());

	std::cout << "Starting actual computation" << std::endl;
	double time = execKernels(kernels, queue);
	std::cout << "Computation done" << std::endl;

	outAdapter.copyFromDeviceAsync(queue);
	OCL_CHECK(err, err = queue.finish());

	return time;
}

enum DataflowType {
	LOCALDATAFLOW, REMOTEDATAFLOW
};

double calc_fpgahbm1kernel(std::vector<vecType, aligned_allocator<vecType>> &vec,std::vector<vecType, aligned_allocator<vecType>> &out,
vecType scale, cl::Context &context, cl::Program &program, cl::Device &device, DataflowType flow) {
	cl_int err;

	assert(vec.size() % (16 * 1024) == 0);
	std::function<unsigned int(unsigned int)> inAdapterFun;
	std::function<unsigned int(unsigned int)> outAdapterFun;
	switch (flow)
	{
	case LOCALDATAFLOW:
		inAdapterFun = [](unsigned int i) {
			return 2*i;
		};
		outAdapterFun = [](unsigned int i) {
			return 2*i+1;
		};
		break;
	case REMOTEDATAFLOW:
		inAdapterFun = [](unsigned int i) {
			return i;
		};
		outAdapterFun = [](unsigned int i) {
			return i + 16;
		};
		break;
	default:
		return -1;
		break;
	}

	MemorySplitHBMAdapter<vecType> inAdapter(inAdapterFun, 16, vec, context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY);
	MemorySplitHBMAdapter<vecType> outAdapter(outAdapterFun,16, out, context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_WRITE_ONLY);
	
	OCL_CHECK(err, cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
	inAdapter.copyToDeviceAsync(queue);
	std::vector<cl::Kernel> kernels(1);
	OCL_CHECK(err, kernels[0] = cl::Kernel(program, "scal", &err));

	unsigned long long xsize = vec.size() / 16;
	OCL_CHECK(err, err= kernels[0].setArg(0, xsize));
	OCL_CHECK(err, err= kernels[0].setArg(1, scale));
	OCL_CHECK(err, err= kernels[0].setArg(2, inAdapter.getBuffer(0)));
	OCL_CHECK(err, err= kernels[0].setArg(3, inAdapter.getBuffer(1)));
	OCL_CHECK(err, err= kernels[0].setArg(4, inAdapter.getBuffer(2)));
	OCL_CHECK(err, err= kernels[0].setArg(5, inAdapter.getBuffer(3)));
	OCL_CHECK(err, err= kernels[0].setArg(6, inAdapter.getBuffer(4)));
	OCL_CHECK(err, err= kernels[0].setArg(7, inAdapter.getBuffer(5)));
	OCL_CHECK(err, err= kernels[0].setArg(8, inAdapter.getBuffer(6)));
	OCL_CHECK(err, err= kernels[0].setArg(9, inAdapter.getBuffer(7)));
	OCL_CHECK(err, err= kernels[0].setArg(10, inAdapter.getBuffer(8)));
	OCL_CHECK(err, err= kernels[0].setArg(11, inAdapter.getBuffer(9)));
	OCL_CHECK(err, err= kernels[0].setArg(12, inAdapter.getBuffer(10)));
	OCL_CHECK(err, err= kernels[0].setArg(13, inAdapter.getBuffer(11)));
	OCL_CHECK(err, err= kernels[0].setArg(14, inAdapter.getBuffer(12)));
	OCL_CHECK(err, err= kernels[0].setArg(15, inAdapter.getBuffer(13)));
	OCL_CHECK(err, err= kernels[0].setArg(16, inAdapter.getBuffer(14)));
	OCL_CHECK(err, err= kernels[0].setArg(17, inAdapter.getBuffer(15)));
	OCL_CHECK(err, err= kernels[0].setArg(18, outAdapter.getBuffer(0)));
	OCL_CHECK(err, err= kernels[0].setArg(19, outAdapter.getBuffer(1)));
	OCL_CHECK(err, err= kernels[0].setArg(20, outAdapter.getBuffer(2)));
	OCL_CHECK(err, err= kernels[0].setArg(21, outAdapter.getBuffer(3)));
	OCL_CHECK(err, err= kernels[0].setArg(22, outAdapter.getBuffer(4)));
	OCL_CHECK(err, err= kernels[0].setArg(23, outAdapter.getBuffer(5)));
	OCL_CHECK(err, err= kernels[0].setArg(24, outAdapter.getBuffer(6)));
	OCL_CHECK(err, err= kernels[0].setArg(25, outAdapter.getBuffer(7)));
	OCL_CHECK(err, err= kernels[0].setArg(26, outAdapter.getBuffer(8)));
	OCL_CHECK(err, err= kernels[0].setArg(27, outAdapter.getBuffer(9)));
	OCL_CHECK(err, err= kernels[0].setArg(28, outAdapter.getBuffer(10)));
	OCL_CHECK(err, err= kernels[0].setArg(29, outAdapter.getBuffer(11)));
	OCL_CHECK(err, err= kernels[0].setArg(30, outAdapter.getBuffer(12)));
	OCL_CHECK(err, err= kernels[0].setArg(31, outAdapter.getBuffer(13)));
	OCL_CHECK(err, err= kernels[0].setArg(32, outAdapter.getBuffer(14)));
	OCL_CHECK(err, err= kernels[0].setArg(33, outAdapter.getBuffer(15)));

	OCL_CHECK(err, err = queue.finish());

	double time = execKernels(kernels, queue);
	
	outAdapter.copyFromDeviceAsync(queue);
	OCL_CHECK(err, err = queue.finish());

	return time;
}

double calc_fpgahbm2kernel(std::vector<vecType, aligned_allocator<vecType>> &vec,std::vector<vecType, aligned_allocator<vecType>> &out,
vecType scale, cl::Context &context, cl::Program &program, cl::Device &device) {
	cl_int err;

	assert(vec.size() % (16 * 1024) == 0);

	MemorySplitHBMAdapter<vecType> inAdapter([](unsigned int i){return 2*i;},
		16, vec, context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY);
	MemorySplitHBMAdapter<vecType> outAdapter([](unsigned int i) {return 2*i+1;},
		16, out, context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_WRITE_ONLY);

	OCL_CHECK(err, cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
	inAdapter.copyToDeviceAsync(queue);
	std::vector<cl::Kernel> kernels(2);

	OCL_CHECK(err, kernels[0] = cl::Kernel(program, "scal:{scal_1}", &err));
	OCL_CHECK(err, kernels[1] = cl::Kernel(program, "scal:{scal_2}", &err));

	for(size_t i = 0; i < 2; i++) {
		unsigned long long xsize = inAdapter.getSplitBlock(0).size();
		OCL_CHECK(err, err= kernels[i].setArg(0, xsize));
		OCL_CHECK(err, err= kernels[i].setArg(1, scale));
		OCL_CHECK(err, err= kernels[i].setArg(2, inAdapter.getBuffer(i*8+0)));
		OCL_CHECK(err, err= kernels[i].setArg(3, inAdapter.getBuffer(i*8+1)));
		OCL_CHECK(err, err= kernels[i].setArg(4, inAdapter.getBuffer(i*8+2)));
		OCL_CHECK(err, err= kernels[i].setArg(5, inAdapter.getBuffer(i*8+3)));
		OCL_CHECK(err, err= kernels[i].setArg(6, inAdapter.getBuffer(i*8+4)));
		OCL_CHECK(err, err= kernels[i].setArg(7, inAdapter.getBuffer(i*8+5)));
		OCL_CHECK(err, err= kernels[i].setArg(8, inAdapter.getBuffer(i*8+6)));
		OCL_CHECK(err, err= kernels[i].setArg(9, inAdapter.getBuffer(i*8+7)));
		OCL_CHECK(err, err= kernels[i].setArg(10, outAdapter.getBuffer(i*8+0)));
		OCL_CHECK(err, err= kernels[i].setArg(11, outAdapter.getBuffer(i*8+1)));
		OCL_CHECK(err, err= kernels[i].setArg(12, outAdapter.getBuffer(i*8+2)));
		OCL_CHECK(err, err= kernels[i].setArg(13, outAdapter.getBuffer(i*8+3)));
		OCL_CHECK(err, err= kernels[i].setArg(14, outAdapter.getBuffer(i*8+4)));
		OCL_CHECK(err, err= kernels[i].setArg(15, outAdapter.getBuffer(i*8+5)));
		OCL_CHECK(err, err= kernels[i].setArg(16, outAdapter.getBuffer(i*8+6)));
		OCL_CHECK(err, err= kernels[i].setArg(17, outAdapter.getBuffer(i*8+7)));
	}
	OCL_CHECK(err, err = queue.finish());

	double time = execKernels(kernels, queue);
	
	outAdapter.copyFromDeviceAsync(queue);
	OCL_CHECK(err, err = queue.finish());

	return time;
}