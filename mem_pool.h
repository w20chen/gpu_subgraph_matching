#ifndef MEM_POOL_H
#define MEM_POOL_H


#include <cstdint>
#include <assert.h>
#include <cstdlib>
#include "helper.h"


class MemPool {
    int *head;              // starting address of memory pool
    int *nextAddrBound;     // upper bound of nextAddr

    union {
        int *val_int;
        unsigned long long val_ull; 
    } 
    nextAddr;               // address of next available block

public:
    const int blockSize = 1024;                     // # of bytes within a block
    const int blockNum = 1024 * 16;                          // # of memory blocks within a mempool
    const int blockIntNum = blockSize / sizeof(int);    // # of integers within a block
    const int poolSize = blockNum * blockSize;          // # of bytes within a mempool

    MemPool() {
        assert(sizeof(char) == 1);
        assert(sizeof(int) == 4);
        assert(sizeof(void *) == 8);
        assert(sizeof(unsigned long long) == 8);
        assert(sizeof(int *) == sizeof(unsigned long long));

        CHECK(cudaMalloc(&head, poolSize));
        CHECK(cudaMemset(head, -1, poolSize));

        nextAddr.val_int = head;
        nextAddrBound = head + blockNum * blockIntNum;
        printf("nextAddrBound: %p\n", nextAddrBound);       // something like 0x7f1de1df4000
    }

    __device__ __forceinline__ int *alloc() {
        if (nextAddr.val_int >= nextAddrBound) {
            printf("No more available block in mempool. nextAddrBound: %p, nextAddr: %p\n",
                nextAddrBound, nextAddr.val_int);
            assert(0);
            return nullptr;
        }

        unsigned long long oldNextAddr = atomicAdd(&nextAddr.val_ull, (unsigned long long)blockSize);
        // printf("Device alloc: %p\n", (int *)oldNextAddr);   // something like 0x7f1de1c00000
        return (int *)oldNextAddr;
    }

    __host__ int *h_alloc() {
        // printf("Host alloc: %p\n", nextAddr.val_int);
        int *addr = nextAddr.val_int;
        nextAddr.val_int += blockIntNum;
        return addr;
    }

    __host__ void freeAll() {
        nextAddr.val_int = head;
    }

    __device__ void print_meta() {
        assert(0);
    }

    void deallocate() {
        CHECK(cudaFree(head));
    }
};

#endif