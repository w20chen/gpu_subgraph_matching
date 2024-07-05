#ifndef MEM_POOL_H
#define MEM_POOL_H


#include <cstdint>
#include <assert.h>
#include <cstdlib>
#include "helper.h"


class MemPool {
    void *head;

    int **freeList;
    bool *isFree;
    int listBack;

    int lock;       // 0,1

public:
    const int blockSize = 4 * 1024;         // # of bytes
    const int blockIntNum = blockSize / 4;  // # of int
    const int blockNum = 200;

    MemPool() {
        lock = 0;

        assert(sizeof(char) == 1);
        assert(sizeof(int) == 4);
        assert(sizeof(void *) == 8);

        CHECK(cudaMalloc(&head, blockNum * blockSize));
        CHECK(cudaMalloc(&freeList, sizeof(int *) * blockNum));
        CHECK(cudaMalloc(&isFree, sizeof(bool) * blockNum));

        int **h_freeList = (int **)malloc(sizeof(int *) * blockNum);
        bool *h_isFree = (bool *)malloc(sizeof(bool) * blockNum);

        for (int i = 0; i < blockNum; i++) {
            h_isFree[i] = true;
            h_freeList[i] = (int *)((char *)head + i * blockSize);
        }

        CHECK(cudaMemcpy(freeList, h_freeList, sizeof(int *) * blockNum, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(isFree, h_isFree, sizeof(bool) * blockNum, cudaMemcpyHostToDevice));

        listBack = blockNum;

        // free(h_isFree);
        // free(h_freeList);
    }

    __device__ void lock_mutex() {
        while (atomicCAS(&lock, 0, 1) != 0);
    }

    __device__ void unlock_mutex() {
        atomicExch(&lock, 0);
    }

    __device__ __forceinline__ int *alloc() {
        if (listBack == 0) {
            return nullptr;
        }
        assert(listBack > 0 && listBack <= blockNum);
        int *addr = freeList[listBack - 1];
        listBack -= 1;

        assert(head <= (void *)addr);
        int index = ((char *)addr - (char *)head) / blockSize;
        assert(isFree[index] == true);
        isFree[index] = false;
        return addr;
    }

    __device__ __forceinline__ void free(int *addr) {
        assert(addr != nullptr);
        assert(addr >= head);
        assert((char *)addr < (char *)head + blockNum * blockSize);
        assert(((char *)addr - (char *)head) % blockSize == 0);

        assert(listBack >= 0 && listBack < blockNum);
        freeList[listBack] = addr;
        listBack += 1;

        int index = ((char *)addr - (char *)head) / blockSize;
        assert(isFree[index] == false);
        isFree[index] = true;
    }

    __device__ void print_meta() {
        printf("mempool starts at %p.\n", head);
        printf("num of available blocks: %d.\n", listBack);
    }

    void deallocate() {
        CHECK(cudaFree(head));
        CHECK(cudaFree(isFree));
        CHECK(cudaFree(freeList));
    }
};

#endif