#include "join_bfs.h"
#include "join.h"


void __global__ BFS_Extend(
    const Graph_GPU Q,
    const Graph_GPU G,
    const candidate_graph_GPU cg,
    const MemManager MM,
    int cur_query_vertex,
    int *d_rank
) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;

    // assign partial matching to each warp
    int u = cur_query_vertex;
    int *this_partial_matching = MM.get_partial(warp_id);

    if (this_partial_matching == nullptr) {
        return;
    }

    // find first backward neighbor fuu of u
    assert(Q.d_bknbrs_offset_ != nullptr);
    assert(Q.d_bknbrs_ != nullptr);
    assert(u >= 0 && u < *Q.d_vcount_);
    int o_ = Q.d_bknbrs_offset_[u];
    int fuu = Q.d_bknbrs_[o_];
    assert(fuu >= 0 && fuu < *Q.d_vcount_);
    assert(Q.d_bknbrs_offset_[u] <= Q.d_bknbrs_offset_[u + 1]);
    int fvv = this_partial_matching[d_rank[fuu]];
    int flen = 0;
    int *fset = cg.d_get_candidates(fuu, u, fvv, flen);

    // allocate a memory block for each warp
    int *d_new_head = nullptr;
    if (lane_id == 0) {
        d_new_head = MM.MP.alloc();
    }
    __shfl_sync(0xffffffff, d_new_head, 0, 64);
    assert(d_new_head != nullptr);

    // compute extendable candidate set
    for (int _i = lane_id; _i < flen; _i += warpSize) {
        int v = fset[_i];
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
            int old_prealloc_cnt = atomicAdd(&MM.blk_wcnt[warp_id], 1);
            int idx = old_prealloc_cnt * (partial_matching_len + 1) + partial_matching_len;
            d_new_head[idx] = v;
            for (int i = 0; i < partial_matching_len; i++) {
                idx = old_prealloc_cnt * (partial_matching_len + 1) + i;
                d_new_head[idx] = this_partial_matching[i];
            }
        }
    }

    __syncthreads();

    if (lane_id == 0) {
        partial_props p;
        p.blk_addr = d_new_head;
        p.partial_len = partial_matching_len + 1;
        p.partial_cnt = MM.blk_wcnt[warp_id];
        MM.new_props[warp_id] = p;
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
    int partial_matching_cnt = 0;
    int *d_partial_matchings = set_beginning_partial_matchings(q, g, _cg, matching_order, partial_matching_cnt);

    MemManager MM;
    MM.init(d_partial_matching, 2, partial_matching_cnt);   // block-wise

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
    for (int i = 0; i < h_rank.size(); i++) {
        printf("%d ", h_rank[i]);
    }
    printf("\n");

    for (int partial_matching_len = 2; partial_matching_len < matching_order.size(); partial_matching_len++) {
        printf("partial matching length: %d\n", partial_matching_len);

        int threadBlocks = (partial_matching_cnt + warpsPerBlock - 1) / warpsPerBlock;
        BFS_Extend<<<threadBlocks, threadsPerBlock>>>(Q, G, cg, MM, matching_order[partial_matching_len], d_rank);

        partial_matching_cnt = MM.new_partial_cnt<<<1, 1>>>();
        MM.update();

        CHECK(cudaDeviceSynchronize());
    }

    return partial_matching_cnt;
}
