#pragma once

#include "include.hpp"

template<typename T>
class MemorySplitHBMAdapter {
private:    
    MemorySplit<T> splitter;
    std::vector<cl::Buffer> managed;
    std::vector<cl_mem_ext_ptr_t> banks;

    void _init(std::vector<size_t> &targets, std::vector<T, aligned_allocator<T>> &data, 
    cl::Context &con, cl_mem_flags flags) {
        splitter.split(data, targets.size());
        size_t blockcount = splitter.countBlocks();
        managed.resize(blockcount);
        banks.resize(blockcount);
        cl_int err;

        for(size_t i = 0; i < blockcount; i++) {
            cl_mem_ext_ptr_t bank_in;
            auto currentBlock = splitter.getBlock(i);

            bank_in.obj = currentBlock.data();
            bank_in.param = 0;
            bank_in.flags = targets[i] | XCL_MEM_TOPOLOGY;
            banks[i] = bank_in;

            #ifdef DEBUGVERBOSE
            std::cout << "Setting Target: " << targets[i] << " with data " << currentBlock.data() << std::endl;
            #endif
            
            OCL_CHECK(err, managed[i] = cl::Buffer(con, flags, 
                sizeof(float)*currentBlock.size(), banks.data() + i, &err));
        }
        #ifdef DEBUGVERBOSE
        std::cout << std::endl;
        #endif
    } 
public:

    MemorySplitHBMAdapter(std::vector<size_t> &targets, std::vector<T, aligned_allocator<T>> &data, 
    cl::Context &con, cl_mem_flags flags = CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX) {
        _init(targets, data, con, flags);        
    }

    MemorySplitHBMAdapter(std::function<size_t(size_t)> targets, size_t targetsize, std::vector<T, aligned_allocator<T>> &data, 
    cl::Context &con, cl_mem_flags flags = CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX) {
        std::vector<size_t> tm(targetsize);
        for(size_t i = 0; i < targetsize; i++) {
            tm[i] = targets(i);
        }
        _init(tm, data, con, flags);
    }


    void copyToDeviceAsync(cl::CommandQueue queue) {
        cl_int err;
        size_t blocks = managed.size();
        for(size_t i = 0; i < blocks; i++) {
            OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({managed[i]}, 0));
        }
    }

/*
   void copyToDeviceAsync(cl::CommandQueue queue) {
        cl_int err;
        size_t blocks = managed.size();
        cl::vector<cl::Memory> u;
        for(size_t i = 0; i < blocks; i++) {
            u.push_back(managed[i]);
        }
        for(size_t i = 0; i < blocks; i++) {
            OCL_CHECK(err, err = queue.enqueueMigrateMemObjects(u, 0));
        }
    }

    void copyFromDeviceAsync(cl::CommandQueue queue) {
        cl_int err;
        size_t blocks = managed.size();
        cl::vector<cl::Memory> u;
        for(size_t i = 0; i < blocks; i++) {
            u.push_back(managed[i]);
        }
        OCL_CHECK(err, err = queue.enqueueMigrateMemObjects(u, CL_MIGRATE_MEM_OBJECT_HOST));
    }
    */

   void copyFromDeviceAsync(cl::CommandQueue queue) {
        cl_int err;
        size_t blocks = managed.size();
        for(size_t i = 0; i < blocks; i++) {
            OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({managed[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
        }
    }

    size_t getInstantiatedBlocks() {
        return splitter.countBlocks();
    }

    cl::Buffer &getBuffer(size_t index) {
        return managed[index];
    }

    MemoryBlock<T> getSplitBlock(size_t index) {
        return splitter.getBlock(index);
    }
};