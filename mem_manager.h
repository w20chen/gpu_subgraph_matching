#ifndef MEM_MANAGER_H
#define MEM_MANAGER_H

#include "mem_pool.h"

struct partial_props {
    int *blk_addr;
    int partial_len;
    int partial_cnt;
};

class MemManager {
    MemPool MP;
    int **warp_blk;

public:
    partial_props *props;
    int props_len;

    partial_props *new_props;
    int new_props_len;

    int *blk_ptrs;

    MemManager() {
        props = nullptr;
        props_len = 0;
        new_props = nullptr;
        new_props_len = 0;
        blk_ptrs = nullptr;
    }

    void init(int *head, int *partial_len, int *partial_cnt) {
        
    }

    void alloc(int warpNum) {
        CHECK(cudaMalloc(&warp_blk, sizeof(int *) * warpNum));
        CHECK(cudaMalloc(&new_props, sizeof(partial_props) * warpNum));
        new_props_len = warpNum;
        CHECK(cudaMalloc(&blk_ptrs, sizeof(int) * warpNum));
    }

    __device__ int *get_this_partial_matching(int warp_id) {
        int cnt = 0;
        for (int i = 0; i < props_len; i++) {
            partial_props p = props[i];
            cnt += p.partial_cnt;
            if (cnt > warp_id) {
                return p.blk_addr + (warp_id - cnt + p.partial_cnt) * p.len * sizeof(int);
            }
        }
        return nullptr;
    }

    __device__ bool blk_valid() {
        return true;
    }

    void reset() {
        CHECK(cudaFree(props));
        CHECK(cudaFree(blk_ptrs));
        props = new_props;
        props_len = new_props_len;
    }

    __device__ int *get_blk(int warp_id) {
        return warp_blk[warp_id];
    }

    __device__ int *alloc_blk(int warp_id) {
        int *blk_addr = alloc();
        warp_blk[warp_id] = blk_addr;
        return blk_addr;
    }
};

#endif