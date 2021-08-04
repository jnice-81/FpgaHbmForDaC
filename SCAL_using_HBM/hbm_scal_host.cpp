#include "include/include.hpp"


int main(int argc, char**argv) {
    cl::Context context;
    cl::Program program;
    cl::Device device;
    cl::Kernel kernel;
    cl::CommandQueue queue;
	size_t vsize;
	vecType scale;
	
	std::string simplemode = "simple", 
		hbmnoburst16pmode = "hbmnoburst16p16",
		hbm1k16p16mode = "hbm1k16p16", 
		hbm4k16p16mode = "hbm4k16p16",
		hbm1k16p1locmode = "hbm1k16p1loc",
		hbm1k16p1remmode = "hbm1k16p1rem",
		hbm1k16p2mode = "hbm1k16p2";
	std::string outputtest = "test", outputverbose = "verbose";

	if (argc < 3) {
        std::cout << "Usage: <action: " << simplemode << "/" << hbmnoburst16pmode << "/" <<
		 hbm1k16p16mode << "/" << hbm4k16p16mode << "/" << hbm1k16p1locmode 
		 << "/" << hbm1k16p1remmode << "/" << hbm1k16p2mode
		 << "> <xclbin> <test/verbose>" << std::endl;
        return -1;
    }
	std::string mode = argv[1];
	std::string outputmode = argv[3];

	loadu280fpga(device, context);
	applyProgram(device, context, program, argv[2]);

	vsize = DATA_SIZE;
	std::srand(1);

	//No reserving memory via new, because it seems xrt expects memory to be aligned to pages
	std::vector<vecType, aligned_allocator<vecType>> vec(vsize);
	auto genfun = [](){
		return std::rand() / (RAND_MAX / 1000.0);
	};
	std::generate(&vec[0], &vec[vsize], genfun);
	//posix_memalign((void **)&out, 4096, vsize * sizeof(vecType));
	std::vector<vecType, aligned_allocator<vecType>> out(vsize);
	scale = genfun();

	std::cout << "Successfully setup environment" << std::endl;

	double timeneeded = 1.0;
	assert(vec.size() == out.size());
	if(mode == simplemode) {
		timeneeded = calc_fpga(vec, out, scale, context, program, device);
	}
	else if(mode == hbmnoburst16pmode) {
		timeneeded = calc_fpgahbm16kernel(vec, out, scale, context, program, device);
	}
	else if(mode == hbm1k16p16mode) {
		timeneeded = calc_fpgahbm16kernel(vec, out, scale, context, program, device);
	}
	else if(mode == hbm4k16p16mode) {
		timeneeded = calc_fpgahbm16kernel(vec, out, scale, context, program, device);
	}
	else if(mode == hbm1k16p1locmode) {
		timeneeded = calc_fpgahbm1kernel(vec, out, scale, context, program, device, LOCALDATAFLOW);
	}
	else if(mode == hbm1k16p1remmode) {
		timeneeded = calc_fpgahbm1kernel(vec, out, scale, context, program, device, REMOTEDATAFLOW);
	}
	else if(mode == hbm1k16p2mode) {
		timeneeded = calc_fpgahbm2kernel(vec, out, scale, context, program, device);
	}
	else {
		std::cout << "Mode not found" << std::endl;
		return -1;
	}

	#ifdef DEBUG
	verifycorrect(vec, scale, out);
	#endif

	double GBs = (vec.size() * sizeof(vecType)) / (timeneeded * 1024*1024*1024);

	if(outputverbose == outputmode) {
		std::cout << std::endl << "Kernel time without time to load/unload data: " << timeneeded << " s" << std::endl
			<< "GB/s computed: " << GBs << std::endl
			<< "Hence memory access speed is " << 2*GBs << "GB's for this program" << std::endl;
	}
	else {
		std::cerr << GBs << std::endl;	//Do it via stderr, because the normal stdout is somewhat crowded during debug
	}
	
	return 0;
}
