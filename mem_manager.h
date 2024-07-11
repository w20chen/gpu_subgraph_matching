#include "mem_pool.h"
#include "helper.h"

struct partial_props {
    int *start_addr;
    int partial_len;
    int partial_cnt;
};

class MemManager {
    partial_props *first_props_array;
    partial_props *second_props_array;

    int first_props_array_len;
    int second_props_array_len;

    MemPool first_mem_pool;
    MemPool second_mem_pool;

    int current_props_array_id;     // ID of MemPool with unfinished partial matchings

public:

    int blockIntNum;                // same in MemPool. # of integers in a memory block

    MemManager() {
        first_props_array = nullptr;
        second_props_array = nullptr;

        first_props_array_len = 0;
        second_props_array_len = 0;

        current_props_array_id = 0;

        blockIntNum = first_mem_pool.blockIntNum;
    }

    void init(int *partial_matching_addr, int partial_matching_cnt, int partial_matching_len = 2) {
        current_props_array_id = 1;

        int partial_matching_num_per_blk = blockIntNum / partial_matching_len;
        int need_blk_num = ceil_div(partial_matching_cnt, partial_matching_num_per_blk);

        partial_props *h_first_props_array = (partial_props *)malloc(sizeof(partial_props) * need_blk_num);
        assert(h_first_props_array != nullptr);

        for (int i = 0; i < need_blk_num; i++) {
            h_first_props_array[i].start_addr = first_mem_pool.h_alloc();
            h_first_props_array[i].partial_cnt = partial_matching_num_per_blk;
            h_first_props_array[i].partial_len = partial_matching_len;
            CHECK(
                cudaMemcpy(
                    h_first_props_array[i].start_addr, 
                    partial_matching_addr + i * partial_matching_num_per_blk * partial_matching_len, 
                    sizeof(int) * blockIntNum, 
                    cudaMemcpyDeviceToDevice
                )
            );
        }

        for (int i = 0; i < need_blk_num; i++) {
            printf("props[%d], start %p, cnt %d, len %d\n", i, h_first_props_array[i].start_addr,
                h_first_props_array[i].partial_cnt, h_first_props_array[i].partial_len);
        }

        // CHECK(cudaMalloc(&first_props_array, sizeof(partial_props) * need_blk_num));
        CHECK(cudaMalloc(&first_props_array, sizeof(partial_props) * 1024));
        CHECK(cudaMalloc(&second_props_array, sizeof(partial_props) * 1024));

        CHECK(cudaMemcpy(first_props_array, h_first_props_array, sizeof(partial_props) * need_blk_num, cudaMemcpyHostToDevice));

        free(h_first_props_array);

        first_props_array_len = need_blk_num;
        second_props_array_len = 0;

        printf("Memory manager initialized.\n");
    }

    __device__ MemPool *write_mempool() {
        if (current_props_array_id == 1) {
            return &second_mem_pool;
        }
        else if (current_props_array_id == 2) {
            return &first_mem_pool;
        }
        else assert(0);
    }

    void swap_mem_pool() {
        if (current_props_array_id == 1) {
            current_props_array_id = 2;
            first_mem_pool.freeAll();
            first_props_array_len = 0;
        }
        else if (current_props_array_id == 2) {
            current_props_array_id = 1;
            second_mem_pool.freeAll();
            second_props_array_len = 0;
        }
        else assert(0);
    }

    __device__ int *get_partial(int warp_id) {
        partial_props *props = nullptr;
        int props_len = 0;
        if (current_props_array_id == 1) {
            props = first_props_array;
            props_len = first_props_array_len;
        }
        else if (current_props_array_id == 2) {
            props = second_props_array;
            props_len = second_props_array_len;
        }
        else assert(0);

        int cnt = 0;
        for (int i = 0; i < props_len; i++) {
            partial_props p = props[i];
            cnt += p.partial_cnt;
            if (cnt > warp_id) {
                return p.start_addr + (warp_id - cnt + p.partial_cnt) * p.partial_len;
            }
        }
        return nullptr;
    }

    __device__ partial_props *get_partial_props(int warp_id) {
        if (current_props_array_id == 1) {
            return first_props_array + warp_id;
        }
        else if (current_props_array_id == 2) {
            return second_props_array + warp_id;
        }
        else assert(0);
    }

    __device__ void add_new_props(partial_props props) {
        if (current_props_array_id == 1) {
            second_props_array[second_props_array_len] = props;
            second_props_array_len += 1;
        }
        else if (current_props_array_id == 2) {
            first_props_array[first_props_array_len] = props;
            first_props_array_len += 1;
        }
        else assert(0);
    }

    __host__ int get_partial_cnt() {
        if (current_props_array_id == 1) {
            partial_props *h_props = (partial_props *)malloc(sizeof(partial_props) * first_props_array_len);
            assert(h_props);
            CHECK(cudaMemcpy(h_props, first_props_array, sizeof(partial_props) * first_props_array_len, cudaMemcpyDeviceToHost));

            int ret = 0;
            for (int i = 0; i < first_props_array_len; i++) {
                ret += h_props[i].partial_cnt;
            }
            free(h_props);
            return ret;
        }
        else if (current_props_array_id == 2) {
            partial_props *h_props = (partial_props *)malloc(sizeof(partial_props) * second_props_array_len);
            assert(h_props);
            CHECK(cudaMemcpy(h_props, second_props_array, sizeof(partial_props) * second_props_array_len, cudaMemcpyDeviceToHost));

            int ret = 0;
            for (int i = 0; i < second_props_array_len; i++) {
                ret += h_props[i].partial_cnt;
            }
            free(h_props);
            return ret;
        }
        else assert(0);
        return 0;
    }
};