#ifndef MEM_POOL_H
#define MEM_POOL_H


#include <cstdint>
#include <assert.h>
#include <cstdlib>
#include "helper.h"


#define memPoolBlockSize 1024                                   // # of bytes within a block
#define memPoolBlockNum 1024 * 16                               // # of memory blocks within a mempool
#define memPoolBlockIntNum (memPoolBlockSize / sizeof(int))     // # of integers within a block


class MemPool {
    int *head;              // starting address of memory pool
    int *nextAddrBound;     // upper bound of nextAddr
    int *nextAddr;          // address of next available block

public:
    MemPool() {
        assert(sizeof(char) == 1);
        assert(sizeof(int) == 4);
        assert(sizeof(void *) == 8);
        assert(sizeof(unsigned long long) == 8);
        assert(sizeof(int *) == sizeof(unsigned long long));

        CHECK(cudaMalloc(&head, memPoolBlockNum * memPoolBlockSize));
        CHECK(cudaMemset(head, -1, memPoolBlockNum * memPoolBlockSize));

        nextAddr = head;
        nextAddrBound = head + memPoolBlockNum * memPoolBlockIntNum;
        printf("head: %p, nextAddrBound: %p\n", head, nextAddrBound);
    }

    __device__ __forceinline__ int *alloc(int flag = 0) {
        if (nextAddr >= nextAddrBound) {
            printf("No more available block in mempool. nextAddrBound: %p, nextAddr: %p\n",
                nextAddrBound, nextAddr);
            assert(0);
            return nullptr;
        }

        unsigned long long oldNextAddr = atomicAdd((unsigned long long *)&nextAddr, (unsigned long long)memPoolBlockSize);
        if (flag) printf("#Device alloc: %p\n", (int *)oldNextAddr);
        else printf("Device alloc: %p\n", (int *)oldNextAddr);
        return (int *)oldNextAddr;
    }

    __host__ int *h_alloc() {
        // printf("Host alloc: %p\n", nextAddr);
        int *addr = nextAddr;
        nextAddr += memPoolBlockIntNum;
        return addr;
    }

    __host__ void freeAll() {
        nextAddr = head;
    }

    void deallocate() {
        CHECK(cudaFree(head));
    }

    static void print_meta() {
        printf("memPoolBlockSize=%d, memPoolBlockNum=%d, memPoolBlockIntNum=%d.\n",
            memPoolBlockSize, memPoolBlockNum, memPoolBlockIntNum);
    }
};

#endif