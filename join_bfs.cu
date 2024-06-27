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

    int u = cur_query_vertex;
    int *this_partial_matching = MM.get_this_partial_matching(warp_id);

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

    int *d_new_head = nullptr;

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
            if (lane_id == 0) {
                if (MM.blk_valid(warp_id)) {
                    d_new_head = MM.get_blk(warp_id);
                }
                else {
                    d_new_head = MM.alloc_blk(warp_id);
                }
            }

            __syncwarp();

            int old_prealloc_cnt = atomicAdd(&d_prealloc_cnt, 1);
            int idx = old_prealloc_cnt * (partial_matching_len + 1) + partial_matching_len;
            d_new_head[idx] = v;
            for (int i = 0; i < partial_matching_len; i++) {
                idx = old_prealloc_cnt * (partial_matching_len + 1) + i;
                d_new_head[idx] = this_partial_matching[i];
            }
        }
    }
    MM.save_new(d_new_head, partial_matching_len);
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
        print_partial_results<<<1, 1>>>(d_partial_matchings, partial_matching_len, partial_matching_cnt);
        CHECK(cudaDeviceSynchronize());
        int *d_new_partial_matchings = nullptr;
        CHECK(cudaMalloc(&d_new_partial_matchings, 1024 * 1024 * 1024));

        CHECK(cudaMemcpyToSymbol(d_prealloc_cnt, &Zero, sizeof(int)));

        int blocks = (partial_matching_cnt + warpsPerBlock - 1) / warpsPerBlock;
        BFS_Extend<<<blocks, threadsPerBlock>>>(Q, G, cg, partial_matching_cnt, partial_matching_len,
                                                d_partial_matchings, d_new_partial_matchings, matching_order[partial_matching_len],
                                                d_rank);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpyFromSymbol(&partial_matching_cnt, d_prealloc_cnt, sizeof(int)));
        CHECK(cudaFree(d_partial_matchings));
        d_partial_matchings = d_new_partial_matchings;
    }

    return partial_matching_cnt;
}
