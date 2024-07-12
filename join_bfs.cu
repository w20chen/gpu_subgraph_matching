#include "join_bfs.h"


__host__ int *set_beginning_partial_matchings(
    const Graph &Q,
    const Graph &G,
    const candidate_graph &cg,
    const std::vector<int> &matching_order,
    int &cnt
) {
    int u0 = matching_order[0];
    int u1 = matching_order[1];

    assert(Q.is_adjacent(u0, u1));
    std::vector<std::pair<int, int>> h_beginning_partial_matching;
    for (int v0 : cg.cand[u0]) {
        for (int v1 : cg.cand[u1]) {
            if (G.is_adjacent(v0, v1)) {
                h_beginning_partial_matching.emplace_back(v0, v1);
            }
        }
    }

    int beginning_partial_matching_cnt = h_beginning_partial_matching.size();
    printf("Beginning partial matching count: %d\n", beginning_partial_matching_cnt);
    assert(beginning_partial_matching_cnt > 0);

    int *d_dst;
    CHECK(cudaMalloc(&d_dst, sizeof(std::pair<int, int>) * beginning_partial_matching_cnt));
    CHECK(cudaMemcpy(d_dst, h_beginning_partial_matching.data(), sizeof(std::pair<int, int>) * beginning_partial_matching_cnt, cudaMemcpyHostToDevice));
    printf("Beginning partial results moved to GPU.\n");

    cnt = beginning_partial_matching_cnt;

    // Debug
    printf("(%d,%d), (%d,%d), ...\n", h_beginning_partial_matching.at(0).first, h_beginning_partial_matching.at(0).second,
        h_beginning_partial_matching.at(1).first, h_beginning_partial_matching.at(1).second);
    return d_dst;
}


void __global__ BFS_Extend(
    const Graph_GPU Q,
    const Graph_GPU G,
    const candidate_graph_GPU cg,
    MemManager MM,
    int cur_query_vertex,
    int *d_rank
) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int warp_id = tid / warpSize;
    int warp_id_in_blk = threadIdx.x / warpSize;
    int lane_id = tid % warpSize;

    // assign partial matching to each warp
    int u = cur_query_vertex;
    int *this_partial_matching = MM.get_partial(warp_id);
    partial_props *props = MM.get_partial_props(warp_id);
    assert(props);
    int partial_matching_len = props->partial_len;

    if (this_partial_matching == nullptr) {
        return;
    }

    // find first backward neighbor fuu of u
    assert(Q.d_bknbrs_offset_ != nullptr);
    assert(Q.d_bknbrs_ != nullptr);
    int o_ = Q.d_bknbrs_offset_[u];
    int fuu = Q.d_bknbrs_[o_];
    assert(Q.d_bknbrs_offset_[u] <= Q.d_bknbrs_offset_[u + 1]);
    int fvv = this_partial_matching[d_rank[fuu]];
    int flen = 0;
    int *fset = cg.d_get_candidates(fuu, u, fvv, flen);

    // allocate a memory block for each warp
    int *d_new_head = 0;
    unsigned d_new_head_lower = 0;
    unsigned d_new_head_upper = 0;
    if (lane_id == 0) {
        d_new_head = MM.write_mempool()->alloc();
        assert(d_new_head != 0);
        d_new_head_lower = (unsigned)d_new_head;
        // printf("%u %p\n", d_new_head_lower, d_new_head);
        d_new_head_upper = (unsigned)((unsigned long long)d_new_head >> 32);
    }

    d_new_head_lower = __shfl_sync(0xffffffff, d_new_head_lower, 0);
    d_new_head_upper = __shfl_sync(0xffffffff, d_new_head_upper, 0);

    __syncwarp();

    d_new_head = (int *)(((unsigned long long)d_new_head_upper << 32) | (unsigned long long)d_new_head_lower);

    assert(d_new_head_upper != 0);
    assert(d_new_head != nullptr);
    // printf("%p\n", d_new_head);      // something like 0x7f19d3c00000

    // block writing counter for each warp
    extern __shared__ int blk_write_cnt[];
    blk_write_cnt[warp_id_in_blk] = 0;

    // compute extendable candidate set
    for (int ii = lane_id; ii < flen; ii += warpSize) {
        int v = fset[ii];
        bool flag = true;
        // for each backward neighbor uu of u (except fuu)
        for (int ii = Q.d_bknbrs_offset_[u] + 1; ii < Q.d_bknbrs_offset_[u + 1]; ii++) {
            int uu = Q.d_bknbrs_[ii];
            int vv = this_partial_matching[d_rank[uu]];
            int len = 0;
            int *this_set = cg.d_get_candidates(uu, u, vv, len);
            if (!binary_search(this_set, len, v)) {
                flag = false;
                break;
            }
        }
        // if v has not been mapped before
        for (int j = 0; j < partial_matching_len; j++) {
            if (this_partial_matching[j] == v) {
                flag = false;
            }
        }
        // v is good
        if (flag) {
            int old_cnt = atomicAdd(&blk_write_cnt[warp_id_in_blk], 1);
            if ((old_cnt + 1) * (partial_matching_len + 1) > MM.blockIntNum) {
                blk_write_cnt[warp_id_in_blk] = 0;
                assert(0);
            }
            // write the newly found partial matching
            int idx = old_cnt * (partial_matching_len + 1) + partial_matching_len;
            d_new_head[idx] = v;
            for (int i = 0; i < partial_matching_len; i++) {
                idx = old_cnt * (partial_matching_len + 1) + i;
                d_new_head[idx] = this_partial_matching[i];
            }
        }
    }

    __syncthreads();

    if (lane_id == 0) {
        if (blk_write_cnt[warp_id] != 0) {
            partial_props p;
            p.start_addr = d_new_head;
            p.partial_len = partial_matching_len + 1;
            p.partial_cnt = blk_write_cnt[warp_id];
            MM.add_new_props(p);
        }
    }
}


int join_bfs(
    const Graph &q,
    const Graph &g,
    const Graph_GPU &Q,
    const Graph_GPU &G,
    const candidate_graph &_cg,
    const candidate_graph_GPU &cg,
    const std::vector<int> &matching_order
) {
    // Generate a table of candidate edges.
    int partial_matching_cnt = 0;
    int *d_partial_matchings = set_beginning_partial_matchings(q, g, _cg, matching_order, partial_matching_cnt);

    MemManager MM;
    // Move a table of initial partial matchings (candidate edges) to memory pool.
    MM.init(d_partial_matchings, partial_matching_cnt);

    CHECK(cudaFree(d_partial_matchings));

    const int threadsPerBlock = 512;
    const int warpsPerBlock = threadsPerBlock / 32;

    int *d_rank = nullptr;
    std::vector<int> h_rank(matching_order.size());
    for (int v = 0; v < matching_order.size(); v++) {
        for (int i = 0; i < matching_order.size(); i++) {
            if (v == matching_order[i]) {
                h_rank[v] = i;
                break;
            }
        }
    }
    CHECK(cudaMalloc(&d_rank, sizeof(int) * matching_order.size()));
    CHECK(cudaMemcpy(d_rank, h_rank.data(), sizeof(int) * matching_order.size(), cudaMemcpyHostToDevice));
    printf("rank: ");
    print_array(h_rank.data(), h_rank.size());

    for (int partial_matching_len = 2; partial_matching_len < matching_order.size(); partial_matching_len++) {
        printf("partial matching length: %d\n", partial_matching_len);

        int partial_matching_cnt = MM.get_partial_cnt();
        printf("cnt: %d\n", partial_matching_cnt);

        // For each partial matching, assign a warp.
        int threadBlocks = ceil_div(partial_matching_cnt, warpsPerBlock);
        BFS_Extend<<<threadBlocks, threadsPerBlock, warpsPerBlock>>>(Q, G, cg, MM, matching_order[partial_matching_len], d_rank);

        CHECK(cudaDeviceSynchronize());

        MM.swap_mem_pool();
    }

    int ret = MM.get_partial_cnt();
    return ret;
}