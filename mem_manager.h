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

    __host__ void
    init(int *partial_matching_addr, int partial_matching_cnt, int partial_matching_len = 2) {
        current_props_array_id = 1;

        int partial_matching_num_per_blk = blockIntNum / partial_matching_len;
        int need_blk_num = ceil_div(partial_matching_cnt, partial_matching_num_per_blk);

        printf("need block num: %d\n", need_blk_num);
        assert(need_blk_num >= 1);

        partial_props *h_first_props_array = (partial_props *)malloc(sizeof(partial_props) * need_blk_num);
        assert(h_first_props_array != nullptr);

        for (int i = 0; i + 1 < need_blk_num; i++) {
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

        // last block
        int l = partial_matching_cnt - partial_matching_num_per_blk * (need_blk_num - 1);
        h_first_props_array[need_blk_num - 1].start_addr = first_mem_pool.h_alloc();
        h_first_props_array[need_blk_num - 1].partial_cnt = l;
        h_first_props_array[need_blk_num - 1].partial_len = partial_matching_len;
        CHECK(
            cudaMemcpy(
                h_first_props_array[need_blk_num - 1].start_addr, 
                partial_matching_addr + (need_blk_num - 1) * partial_matching_num_per_blk * partial_matching_len, 
                l * partial_matching_len * sizeof(int),
                cudaMemcpyDeviceToDevice
            )
        );

        // Debug
        // printf("#: %d\n", (need_blk_num - 2) * partial_matching_num_per_blk * partial_matching_len);
        // int W[4];
        // CHECK(cudaMemcpy(W, h_first_props_array[need_blk_num - 1].start_addr, sizeof(int) * 4, cudaMemcpyDeviceToHost));
        // // 0x770216006000 --> 24843 3457 25253 587
        // printf("%p --> %d %d %d %d\n", h_first_props_array[need_blk_num - 1].start_addr, W[0], W[1], W[2], W[3]);

        for (int i = 0; i < need_blk_num; i++) {
            printf("props[%d], start %p, cnt %d, len %d\n", i, h_first_props_array[i].start_addr,
                h_first_props_array[i].partial_cnt, h_first_props_array[i].partial_len);
        }

        // CHECK(cudaMalloc(&first_props_array, sizeof(partial_props) * need_blk_num));
        CHECK(cudaMalloc(&first_props_array, sizeof(partial_props) * 1024 * 1024));
        CHECK(cudaMalloc(&second_props_array, sizeof(partial_props) * 1024 * 1024));

        CHECK(cudaMemcpy(first_props_array, h_first_props_array, sizeof(partial_props) * need_blk_num, cudaMemcpyHostToDevice));

        first_props_array_len = need_blk_num;
        second_props_array_len = 0;

        printf("Memory manager initialized.\n");

        // Debug
        // int *A = (int *)malloc(sizeof(int) * 4);
        // int *B = (int *)malloc(sizeof(int) * 4);
        // CHECK(cudaMemcpy(A, h_first_props_array[0].start_addr, sizeof(int) * 4, cudaMemcpyDeviceToHost));
        // CHECK(cudaMemcpy(B, h_first_props_array[1].start_addr, sizeof(int) * 4, cudaMemcpyDeviceToHost));
        // printf("Candidate edges: (%d,%d), (%d,%d), (%d,%d), (%d,%d), ...\n", A[0], A[1], A[2], A[3], B[0], B[1], B[2], B[3]);

        free(h_first_props_array);

        // this->dump("start.txt");
    }

    __device__ MemPool *write_mempool() {
        if (current_props_array_id == 1) {
            return &second_mem_pool;
        }
        else if (current_props_array_id == 2) {
            return &first_mem_pool;
        }
        else assert(0);
        return nullptr;
    }

    __host__ void swap_mem_pool() {
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

    __device__ int *get_partial(int warp_id, int *partial_matching_len) {
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
                *partial_matching_len = p.partial_len;
                int *ret = p.start_addr + (warp_id - cnt + p.partial_cnt) * p.partial_len;
                return ret;
            }
        }
        *partial_matching_len = 0;
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
        return nullptr;
    }

    __device__ void add_new_props(partial_props props) {
        assert(props.partial_cnt > 0);
        if (current_props_array_id == 1) {
            int old = atomicAdd(&second_props_array_len, 1);
            assert(old != second_props_array_len);
            second_props_array[old] = props;
            // printf("second_props_array length: %d\n", second_props_array_len);
        }
        else if (current_props_array_id == 2) {
            int old = atomicAdd(&first_props_array_len, 1);
            first_props_array[old] = props;
            // printf("first_props_array length: %d\n", first_props_array_len);
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
            printf("first_props_array length: %d\n", first_props_array_len);
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
            printf("second_props_array length: %d\n", second_props_array_len);
            return ret;
        }
        else assert(0);
        return 0;
    }

    __host__ void dump(char *filename) {
        FILE *fp = fopen(filename, "w");
        assert(fp != nullptr);
        partial_props *d_props_array = nullptr;
        int props_array_len = 0;
        if (current_props_array_id == 1) {
            d_props_array = first_props_array;
            props_array_len = first_props_array_len;
        }
        else if (current_props_array_id == 2) {
            d_props_array = second_props_array;
            props_array_len = second_props_array_len;
        }
        else assert(0);

        partial_props *props_array = (partial_props *)malloc(sizeof(partial_props) * props_array_len);
        assert(props_array != nullptr);
        CHECK(cudaMemcpy(props_array, d_props_array, sizeof(partial_props) * props_array_len, cudaMemcpyDeviceToHost));

        for (int i = 0; i < props_array_len; i++) {     // for each allocated block
            partial_props *p = props_array + i;
            int num = p->partial_cnt;
            int len = p->partial_len;
            int *d_addr = p->start_addr;
            int *h_addr = (int *)malloc(sizeof(int) * num * len);
            CHECK(cudaMemcpy(h_addr, d_addr, sizeof(int) * num * len, cudaMemcpyDeviceToHost));
            for (int j = 0; j < num; j++) {
                int *line = h_addr + len * j;
                for (int k = 0; k < len; k++) {
                    fprintf(fp, "%d,", line[k]);
                }
                fprintf(fp, "\n");
            }
            free(h_addr);
        }

        fclose(fp);
        printf("Result saved in %s\n", filename);
    }
};
