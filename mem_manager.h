#include "mem_pool.h"
#include "helper.h"

struct partial_props {
    int *start_addr;
    int partial_len;
    int partial_cnt;
};

class MemManager {
    partial_props *_props_array[2];
    int _props_array_len[2];
    MemPool _mem_pool[2];

    int current_props_array_id;     // ID of MemPool with unfinished partial matchings
    const int props_max_num = 1024 * 1024;

public:
    MemManager() {
        _props_array[0] = nullptr;
        _props_array[1] = nullptr;

        _props_array_len[0] = 0;
        _props_array_len[1] = 0;

        current_props_array_id = 0;
    }

    __host__ void init(int *partial_matching_addr, int partial_matching_cnt) {
        current_props_array_id = 0;

        const int partial_matching_num_per_blk = memPoolBlockIntNum / 2;
        assert(memPoolBlockIntNum % 2 == 0);
        int need_blk_num = ceil_div(partial_matching_cnt, partial_matching_num_per_blk);

        printf("need block num: %d\n", need_blk_num);
        assert(need_blk_num >= 1);

        partial_props *h_first_props_array = (partial_props *)malloc(sizeof(partial_props) * need_blk_num);
        assert(h_first_props_array != nullptr);

        int i = 0;
        for (i = 0; i + 1 < need_blk_num; i++) {
            h_first_props_array[i].start_addr = partial_matching_addr + 2 * i * partial_matching_num_per_blk;
            h_first_props_array[i].partial_cnt = partial_matching_num_per_blk;
            h_first_props_array[i].partial_len = 2;
        }

        // last block
        assert(i + 1 == need_blk_num);
        h_first_props_array[i].start_addr = partial_matching_addr + 2 * i * partial_matching_num_per_blk;
        h_first_props_array[i].partial_cnt = partial_matching_cnt - partial_matching_num_per_blk * i;
        h_first_props_array[i].partial_len = 2;

        for (int i = 0; i < need_blk_num; i++) {
            printf("props[%d]: start %p, cnt %d, len %d\n", 
            i, h_first_props_array[i].start_addr,
            h_first_props_array[i].partial_cnt, h_first_props_array[i].partial_len);
        }

        assert(props_max_num >= memPoolBlockNum);
        CHECK(cudaMalloc(&_props_array[0], sizeof(partial_props) * props_max_num));
        CHECK(cudaMalloc(&_props_array[1], sizeof(partial_props) * props_max_num));

        CHECK(cudaMemcpy(_props_array[0], h_first_props_array, sizeof(partial_props) * need_blk_num, cudaMemcpyHostToDevice));

        _props_array_len[0] = need_blk_num;
        _props_array_len[1] = 0;

        printf("Memory manager initialized.\n");
        free(h_first_props_array);

        // this->dump("start.txt");
    }

    __device__ MemPool *mempool_to_write() {
        return _mem_pool + (current_props_array_id ^ 1);
    }

    __host__ void swap_mem_pool() {
        _mem_pool[current_props_array_id].freeAll();
        _props_array_len[current_props_array_id] = 0;
        current_props_array_id ^= 1;
    }

    __device__ int *get_partial(int warp_id, int *partial_matching_len) {
        partial_props *props = _props_array[current_props_array_id];
        int props_len = _props_array_len[current_props_array_id];

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
        return _props_array[current_props_array_id] + warp_id;
    }

    __device__ void add_new_props(partial_props props) {
        assert(props.partial_cnt > 0);
        int old = atomicAdd(&_props_array_len[current_props_array_id ^ 1], 1);
        assert(old >= 0 && old < props_max_num);
        _props_array[current_props_array_id ^ 1][old] = props;
    }

    __host__ int get_partial_cnt() {
        partial_props *h_props = (partial_props *)malloc(sizeof(partial_props) * _props_array_len[current_props_array_id]);
        assert(h_props);
        CHECK(cudaMemcpy(h_props, _props_array[current_props_array_id], sizeof(partial_props) * _props_array_len[current_props_array_id], cudaMemcpyDeviceToHost));

        int ret = 0;
        for (int i = 0; i < _props_array_len[current_props_array_id]; i++) {
            ret += h_props[i].partial_cnt;
        }
        free(h_props);
        return ret;
    }

    __host__ void dump(char *filename) {
        FILE *fp = fopen(filename, "w");
        assert(fp != nullptr);
        int props_array_len = _props_array_len[current_props_array_id];

        partial_props *props_array = (partial_props *)malloc(sizeof(partial_props) * props_array_len);
        assert(props_array != nullptr);
        CHECK(cudaMemcpy(props_array, _props_array[current_props_array_id], sizeof(partial_props) * props_array_len, cudaMemcpyDeviceToHost));

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

    void deallocate() {
        _mem_pool[0].deallocate();
        _mem_pool[1].deallocate();
        CHECK(cudaFree(_props_array[0]));
        CHECK(cudaFree(_props_array[1]));
    }
};
