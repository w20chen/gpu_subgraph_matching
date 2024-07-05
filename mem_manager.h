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

public:
    partial_props *props;
    int props_len;

    partial_props *new_props;
    int new_props_len;

    int *blk_wcnt;

    MemManager() {
        props = nullptr;
        props_len = 0;
        new_props = nullptr;
        new_props_len = 0;
        blk_wcnt = nullptr;
    }

    void init(int *head, int partial_len, int partial_cnt) {
        partial_props *h_props = nullptr;

        int num_per_blk = MP.blockSize / (partial_len * sizeof(int));
        int blk_num = (partial_cnt + num_per_blk - 1) / num_per_blk;
        h_props = (partial_props *)malloc(sizeof(partial_props) * blk_num);

        for (int i = 0; i < blk_num; i++) {
            h_props[i].blk_addr = head + partial_len * i;
            h_props[i].partial_len = partial_len;
            h_props[i].partial_cnt = num_per_blk;
        }
        h_props[blk_num].partial_cnt = partial_cnt - (num_per_blk * blk_num);

        props_len = blk_num;
        CHECK(cudaMalloc(&props, props_len * sizeof(partial_props)));
        CHECK(cudaMemcpy(props, h_props, props_len * sizeof(partial_props)));

        printf("Mempool initialized with beginning partial matchings.\n");
        printf("num of memblk: %d\n", props_len);

        int *h_blk_wcnt = (int *)malloc(sizeof(int) * blk_num);
        for (int i = 0; i < blk_num; i++) {
            h_blk_wcnt[i] = 0;
        }
        CHECK(cudaMalloc(&blk_wcnt, sizeof(int) * blk_num));
        CHECK(cudaMemcpy(blk_wcnt, h_blk_wcnt, sizeof(int) * blk_num));

        free(h_props);
        free(h_blk_wcnt);

        CHECK(cudaMalloc(&props, sizeof(int) * blk_num));
        new_props_len = blk_num;
    }

    __device__ int *get_partial(int warp_id) {
        int cnt = 0;
        for (int i = 0; i < props_len; i++) {
            partial_props p = props[i];
            cnt += p.partial_cnt;
            if (cnt > warp_id) {
                return p.blk_addr + (warp_id - cnt + p.partial_cnt) * p.len;
            }
        }
        return nullptr;
    }

    void update() {
        partial_props *tmp = props;
        props = new_props;
        new_props = tmp;

        // reset blk_wcnt
        int *h_blk_wcnt = (int *)malloc(sizeof(int) * blk_num);
        for (int i = 0; i < blk_num; i++) {
            h_blk_wcnt[i] = 0;
        }
        CHECK(cudaMalloc(&blk_wcnt, sizeof(int) * blk_num));
        CHECK(cudaMemcpy(blk_wcnt, h_blk_wcnt, sizeof(int) * blk_num));
        free(h_blk_wcnt);
    }

    void deallocate() {
        if (blk_wcnt) {
            CHECK(cudaFree(blk_wcnt));
            blk_wcnt = nullptr;
        }
        if (props) {
            CHECK(cudaFree(props));
            props = nullptr;
        }
        if (new_props) {
            CHECK(cudaFree(new_props));
            new_props = nullptr;
        }
    }

    int new_partial_cnt() {
        int blk_num = props_len;

        int *h_blk_wcnt = (int *)malloc(sizeof(int) * blk_num);
        CHECK(cudaMemcpy(h_blk_wcnt, blk_wcnt, sizeof(int) * blk_num, cudaMemcpy));

        int cnt = 0;
        for (int i = 0; i < blk_num; i++) {
            cnt += h_blk_wcnt[i];
        }
        free(h_blk_wcnt);
        return cnt;
    }
};

#endif