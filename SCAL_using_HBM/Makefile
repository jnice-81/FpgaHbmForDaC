.PHONY: help

#TODO update this
help::
	@echo "TARGET=<sw_emu, hw_emu, hw> specifies the build target for the kernels"
	@echo "make all: Build host and kernels for x86, platform xilinx_u280_xdma_201920_3"
	@echo "make host: Build host"
	@echo "make clean: remove some of the logs"
	@echo "The kernels-targets follow the naming structure."
	@echo ""

HOST_ARCH := x86
PLATFORM := xilinx_u280_xdma_201920_3
PROJ_ROOT := $(PWD)
BUILD_DIR := $(PWD)/build_scal
BUILD_DIR_RWBENCH := $(PWD)/build_readwritebenchmark
EXECUTABLE := host
INFO_DIR := $(PWD)/info
COMPILE_TIME := $(INFO_DIR)/compile_time.txt
RUN_THROUGHPUT := $(INFO_DIR)/run_time.txt
TARGET := sw_emu

include $(PROJ_ROOT)/common/includes/opencl/opencl.mk
CXXFLAGS += $(opencl_CXXFLAGS) -Wall -std=c++11 -I$(PROJ_ROOT)/common/includes/xcl2

ifneq ($(TARGET), hw)
	CXXFLAGS += -O1 -g 
else
	CXXFLAGS += -O3
endif

LDFLAGS += $(opencl_LDFLAGS) -lrt -lstdc++

HOST_SRC += $(PROJ_ROOT)/common/includes/xcl2/xcl2.cpp ./hbm_scal_host.cpp
KERNEL_SRC = $(PWD)/kernel_scal/
CONFIG_SRC = $(PWD)/config/

CXX := g++
VXX := v++

VXXFLAGS += -t $(TARGET) --platform $(PLATFORM)
ifneq ($(TARGET), hw)
	VXXFLAGS += -g
endif

#Build host
#Note: here $@=$(EXECUTABLE), $^=$(HOST_SRC)
$(EXECUTABLE): $(HOST_SRC)
	$(CXX) -o $@ $(HOST_SRC) $(CXXFLAGS) $(LDFLAGS)

############## Build Kernels

#Build simple kernel

$(BUILD_DIR)/scal_simple_1k_16p_1kernel.xo: $(KERNEL_SRC)/scal_simple_1k_16p_1kernel.cpp
	mkdir -p $(BUILD_DIR)
	$(VXX) $(VXXFLAGS) -c -k scal -o $@ $^

$(BUILD_DIR)/scal_simple_1k_16p_1kernel.xclbin: $(BUILD_DIR)/scal_simple_1k_16p_1kernel.xo
	$(VXX) $(VXXFLAGS) --config $(CONFIG_SRC)/scal_simple.cfg -l -o $@ $^

.PHONY: scal_simple_1k_16p_1kernel
scal_simple_1k_16p_1kernel: $(BUILD_DIR)/scal_simple_1k_16p_1kernel.xclbin

#Build HBM noburst 16p 16 kernels

$(BUILD_DIR)/scal_hbm_noburst_16p_16kernel.xo: $(KERNEL_SRC)/scal_hbm_noburst_16p_16kernel.cpp
	mkdir -p $(BUILD_DIR)
	$(VXX) $(VXXFLAGS) -c -k scal -o $@ $^

$(BUILD_DIR)/scal_hbm_noburst_16p_16kernel.xclbin: $(BUILD_DIR)/scal_hbm_noburst_16p_16kernel.xo
	$(VXX) $(VXXFLAGS) --config $(CONFIG_SRC)/scal_hbm_16kernel_alternate.cfg -l -o $@ $^

.PHONY: scal_hbm_noburst_16p_16kernel
scal_hbm_noburst_16p_16kernel: $(BUILD_DIR)/scal_hbm_noburst_16p_16kernel.xclbin	

#Build HBM 4k 16p 16 kernels

$(BUILD_DIR)/scal_hbm_4k_16p_16kernel.xo: $(KERNEL_SRC)/scal_hbm_4k_16p_16kernel.cpp
	mkdir -p $(BUILD_DIR)
	$(VXX) $(VXXFLAGS) -c -k scal -o $@ $^

$(BUILD_DIR)/scal_hbm_4k_16p_16kernel.xclbin: $(BUILD_DIR)/scal_hbm_4k_16p_16kernel.xo
	$(VXX) $(VXXFLAGS) --config $(CONFIG_SRC)/scal_hbm_16kernel_alternate.cfg -l -o $@ $^

.PHONY: scal_hbm_4k_16p_16kernel
scal_hbm_4k_16p_16kernel: $(BUILD_DIR)/scal_hbm_4k_16p_16kernel.xclbin	

#Build HBM 1k 16p 16 kernels

$(BUILD_DIR)/scal_hbm_1k_16p_16kernel.xo: $(KERNEL_SRC)/scal_hbm_1k_16p_16kernel.cpp
	mkdir -p $(BUILD_DIR)
	$(VXX) $(VXXFLAGS) -c -k scal -o $@ $^ 

$(BUILD_DIR)/scal_hbm_1k_16p_16kernel.xclbin: $(BUILD_DIR)/scal_hbm_1k_16p_16kernel.xo
	$(VXX) $(VXXFLAGS) --config $(CONFIG_SRC)/scal_hbm_16kernel_alternate.cfg -l -o $@ $^

.PHONY: scal_hbm_1k_16p_16kernel
scal_hbm_1k_16p_16kernel: $(BUILD_DIR)/scal_hbm_1k_16p_16kernel.xclbin	


#Build HBM 1k 16p 1 kernel, dataflow only between the same bank

$(BUILD_DIR)/scal_hbm_1k_16p_1kernel.xo: $(KERNEL_SRC)/scal_hbm_1k_16p_1kernel.cpp
	mkdir -p $(BUILD_DIR)
	$(VXX) $(VXXFLAGS) -c -k scal -o $@ $^

$(BUILD_DIR)/scal_hbm_1k_16p_1kernel_localflow.xclbin: $(BUILD_DIR)/scal_hbm_1k_16p_1kernel.xo
	$(VXX) $(VXXFLAGS) --config $(CONFIG_SRC)/scal_hbm_1kernel_bankseperate.cfg -l -o $@ $^

.PHONY: scal_hbm_1k_16p_1kernel_localflow
scal_hbm_1k_16p_1kernel_localflow: $(BUILD_DIR)/scal_hbm_1k_16p_1kernel_localflow.xclbin	

#Build HBM 1k 16p 1 kernel, dataflow only from bank to bank

$(BUILD_DIR)/scal_hbm_1k_16p_1kernel_remoteflow.xclbin: $(BUILD_DIR)/scal_hbm_1k_16p_1kernel.xo
	mkdir -p $(BUILD_DIR)
	$(VXX) $(VXXFLAGS) --config $(CONFIG_SRC)/scal_hbm_1kernel_bank2bank.cfg -l -o $@ $^

.PHONY: scal_hbm_1k_16p_1kernel_remoteflow
scal_hbm_1k_16p_1kernel_remoteflow: $(BUILD_DIR)/scal_hbm_1k_16p_1kernel_remoteflow.xclbin	

#Build HBM 1k 16p 2kernels, only local dataflow
$(BUILD_DIR)/scal_hbm_1k_16p_2kernel.xo: $(KERNEL_SRC)/scal_hbm_1k_16p_2kernel.cpp
	mkdir -p $(BUILD_DIR)
	$(VXX) $(VXXFLAGS) -c -k scal -o $@ $^

$(BUILD_DIR)/scal_hbm_1k_16p_2kernel.xclbin: $(BUILD_DIR)/scal_hbm_1k_16p_2kernel.xo
	$(VXX) $(VXXFLAGS) --config $(CONFIG_SRC)/scal_hbm_2kernel_alternate.cfg -l -o $@ $^

.PHONY: scal_hbm_1k_16p_2kernel
scal_hbm_1k_16p_2kernel: $(BUILD_DIR)/scal_hbm_1k_16p_2kernel.xclbin		

#Helperstuff

.PHONY: all_scal
all_scal:
	make scal_hbm_1k_16p_16kernel
	make scal_hbm_4k_16p_16kernel
	make scal_hbm_1k_16p_2kernel
	make scal_simple_1k_16p_1kernel
	make scal_hbm_1k_16p_1kernel_remoteflow
	make scal_hbm_1k_16p_1kernel_localflow
	make scal_hbm_noburst_16p_16kernel

.PHONY: run_scal
run_scal:
	mkdir -p $(INFO_DIR)
	echo -n "scal_simple_1k_16p_1kernel," >> $(RUN_THROUGHPUT)
	./host simple build_scal/scal_simple_1k_16p_1kernel.xclbin test 2>> $(RUN_THROUGHPUT)
	echo -n "scal_hbm_1k_16p_16kernel," >> $(RUN_THROUGHPUT)
	./host hbm1k16p16 build_scal/scal_hbm_1k_16p_16kernel.xclbin test 2>> $(RUN_THROUGHPUT)
	echo -n "scal_hbm_4k_16p_16kernel," >> $(RUN_THROUGHPUT)
	./host hbm4k16p16 build_scal/scal_hbm_4k_16p_16kernel.xclbin test 2>> $(RUN_THROUGHPUT)
	echo -n "scal_hbm_1k_16p_1kernel_localflow," >> $(RUN_THROUGHPUT)
	./host hbm1k16p1loc build_scal/scal_hbm_1k_16p_1kernel_localflow.xclbin test 2>> $(RUN_THROUGHPUT)
	echo -n "scal_hbm_1k_16p_1kernel_remoteflow," >> $(RUN_THROUGHPUT)
	./host hbm1k16p1rem build_scal/scal_hbm_1k_16p_1kernel_remoteflow.xclbin test 2>> $(RUN_THROUGHPUT)
	echo -n "scal_hbm_1k_16p_2kernel," >> $(RUN_THROUGHPUT)
	./host hbm1k16p2 build_scal/scal_hbm_1k_16p_2kernel.xclbin test 2>> $(RUN_THROUGHPUT)
	echo -n "scal_hbm_noburst_16p_16kernel," >> $(RUN_THROUGHPUT)
	./host hbmnoburst16p16 build_scal/scal_hbm_noburst_16p_16kernel.xclbin test 2>> $(RUN_THROUGHPUT)

.PHONY: clean
clean:
	rm $(PWD)/*.log -f
	rm $(PWD)/.run -r -f
	rm $(PWD)/.ipcache -r -f
	rm $(PWD)/_x -r -f
	rm $(PWD)/.Xil -r -f

.PHONY: cleanhost
cleanhost:
	rm -f $(EXECUTABLE)

.PHONY: cleankernel
cleankernel:
	rm $(BUILD_DIR)/* -f
