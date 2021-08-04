#pragma once

#include "include.hpp"

#define MEMORY_SPLIT_CHUNK_SIZE 4096

template<typename T>
class MemoryBlock {
public:
    MemoryBlock(T* beg, size_t size) {
        this->beg = beg;
        this->sizeblock = size;
    }

    constexpr T *data() {
        return beg;
    }

    constexpr size_t size() {
        return sizeblock;
    }

private:
    T* beg;
    size_t sizeblock;
};

/*
Divides a contigous block of memory into comp_dev blocks,
where each block is a multiple of MEMORY_SPLIT_CHUNK_SIZE (in bytes) except the last one.
If the block is smaller than comp_dev * MEMORY_SPLIT_CHUNK_SIZE, it will create less blocks.
Note that it is necessary that MEMORY_SPLIT_CHUNK_SIZE is dividable by sizeof(T)

Obviously vector must not be modified after creating this. (Or split must be called again)
*/
template<typename T>
class MemorySplit {
public:
    MemorySplit() {
        assert(MEMORY_SPLIT_CHUNK_SIZE % sizeof(T) == 0);
    }

	MemorySplit(std::vector<T, aligned_allocator<T>> &vec, const size_t &comp_dev) {
        assert(MEMORY_SPLIT_CHUNK_SIZE % sizeof(T) == 0);
		split(vec, comp_dev);
    }

    void split(std::vector<T, aligned_allocator<T>> &vec, const size_t &comp_dev) {
        std::vector<size_t> perDevice(comp_dev);
        begin.reserve(comp_dev);
        blocksize.reserve(comp_dev);
        begin.resize(0);
        blocksize.resize(0);

        const size_t totalsize = vec.size();
        const size_t localChunkSize = MEMORY_SPLIT_CHUNK_SIZE / sizeof(T);
        const size_t chunks_down = totalsize / localChunkSize;
        const size_t partsize = chunks_down / comp_dev;
	    const size_t numadd1 = chunks_down % comp_dev;
        T *ptr = vec.data();
        for(size_t i = 0; i < comp_dev; i++) {
            size_t current = partsize;
            if(i < numadd1) {
			    current++;
		    }
            current = current * localChunkSize;
            if(current > 0 || (i == comp_dev - 1 && totalsize % localChunkSize > 0)) {
                begin.push_back(ptr);
                blocksize.push_back(current);
                ptr += current;
			}
        }
        blocksize.back() += totalsize % localChunkSize;

		#ifdef DEBUGVERBOSE

		std::cout << "Created, size: "  << countBlocks() << " start at " << vec.data() << " len " << vec.size() * sizeof(T) << std::endl;
		for(size_t i = 0; i < countBlocks(); i++) {
			std::cout << "Block " << i << ": (" << begin[i] << " , " << blocksize[i] << ")" << std::endl;
		}
        std::cout << std::endl;
		#endif
    }

    size_t countBlocks() {
        return begin.size();
    }

    MemoryBlock<T> getBlock(size_t index) {
        return MemoryBlock<T>(begin[index], blocksize[index]);
    }
private:
    std::vector<T*> begin;
    std::vector<size_t> blocksize;
};

