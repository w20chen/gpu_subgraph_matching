#ifndef MEM_POOL_H
#define MEM_POOL_H


#include <cstdint>
#include <assert.h>
#include <cstdlib>
#include "helper.h"


class MemPool {
    int *head;              // starting address of memory pool
    int *nextAddr;          // address of next available block
    int *nextAddrBound;     // upper bound of nextAddr
    int lock;               // 0, 1

public:
    const int blockSize = 4 * 1024;                     // # of bytes within a block
    const int blockNum = 500;                           // # of memory blocks
    const int blockIntNum = blockSize / sizeof(int);    // # of integers within a block
    const int poolSize = blockNum * blockSize;          // # of bytes within a mempool

    MemPool() {
        lock = 0;

        assert(sizeof(char) == 1);
        assert(sizeof(int) == 4);
        assert(sizeof(void *) == 8);
        assert(sizeof(unsigned long long) == 8);

        CHECK(cudaMalloc(&head, poolSize));
        CHECK(cudaMemset(head, -1, poolSize));

        nextAddr = head;
        nextAddrBound = head + blockNum * blockIntNum;
        // printf("nextAddrBound: %p\n", nextAddrBound);    // something like 0x7f1de1df4000
    }

    __device__ __forceinline__ int *alloc() {
        // printf("nextAddr: %p\n", nextAddr);              // something like 0x7f1de1c00000
        if (nextAddr >= nextAddrBound) {
            printf("No more available block in mempool\n");
            assert(0);
            return nullptr;
        }

        unsigned long long ull_nextAddr = (unsigned long long)nextAddr;
        unsigned long long ull_addr = atomicAdd(&ull_nextAddr, (unsigned long long)blockSize);

        return (int *)ull_addr;
    }

    __host__ int *h_alloc() {
        int *addr = nextAddr;
        nextAddr += blockIntNum;
        return addr;
    }

    __host__ void freeAll() {
        nextAddr = head;
    }

    __device__ void print_meta() {
        assert(0);
    }

    void deallocate() {
        CHECK(cudaFree(head));
    }
};

#endif