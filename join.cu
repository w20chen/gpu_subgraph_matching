#include "join.h"

__device__ int d_prealloc_cnt1;

// ull DFS_CPU(
//     const Graph &Q,
//     const Graph &G,
//     const std::vector<int> &matching_order,
//     const std::vector<std::vector<int>> &cand,
//     int *rank
// ) {

//     if (dfs_stk.size() == Q.vcount()) {
//         return 1;
//     }

//     int partial_matching_length = dfs_stk.size();
//     int u = matching_order[partial_matching_length];
//     ull ret = 0;
//     for (int v : cand[u]) {
//         bool flag = true;
//         for (int uu : Q.adj_[u]) {
//             if (rank[uu] < rank[u]) {
//                 assert(rank[uu] < dfs_stk.size());
//                 int vv = dfs_stk[rank[uu]];
//                 if (!G.is_adjacent(v, vv)) {
//                     flag = false;
//                     break;
//                 }
//             }
//         }
//         if (flag) {
//             for (int w : dfs_stk) {
//                 if (w == v) {
//                     flag = false;
//                     break;
//                 }
//             }
//         }
//         if (flag) {
//             dfs_stk.emplace_back(v);
//             ret += DFS_CPU(Q, G, matching_order, cand, rank);
//             dfs_stk.pop_back();
//         }
//     }
//     return ret;
// }

int *set_beginning_partial_matchings(
    const Graph &Q,
    const Graph &G,
    const candidate_graph &cg,
    const std::vector<int> &matching_order,
    int &cnt
) {
    int u0 = matching_order[0];
    int u1 = matching_order[1];

    assert(Q.is_adjacent(u0, u1));
    std::vector<pint> h_beginning_partial_matching;
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
    CHECK(cudaMalloc(&d_dst, sizeof(pint) * beginning_partial_matching_cnt));
    CHECK(cudaMemcpy(d_dst, h_beginning_partial_matching.data(), sizeof(pint) * beginning_partial_matching_cnt, cudaMemcpyHostToDevice));
    printf("Beginning partial results moved to GPU.\n");

    cnt = beginning_partial_matching_cnt;
    return d_dst;
}

void __global__ ExtendKernel(
    const Graph_GPU &Q,
    const Graph_GPU &G,
    const candidate_graph_GPU &cg,
    int partial_matching_cnt,
    int partial_matching_len,
    int *d_head,
    int *d_new_head,
    int cur_query_vertex,
    int *d_rank
) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;

    int u = cur_query_vertex;
    int *this_partial_matching = d_head + partial_matching_len * warp_id;

    if (warp_id >= partial_matching_cnt) {
        return;
    }

    int fuu = Q.d_bknbrs_[Q.d_bknbrs_offset_[u]];
    assert(Q.d_bknbrs_offset_[u] <= Q.d_bknbrs_offset_[u + 1]);
    int fvv = this_partial_matching[d_rank[fuu]];
    int flen = 0;
    int *fset = cg.d_get_candidates(fuu, u, fvv, flen);

    for (int _i = lane_id; _i < flen; _i += warpSize) {
        int v = fset[_i];
        bool flag = true;
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
        for (int j = 0; j < partial_matching_len; j++) {
            if (this_partial_matching[j] == v) {
                flag = false;
            }
        }
        if (flag) {
            int old_prealloc_cnt1 = atomicAdd(&d_prealloc_cnt1, 1);
            int idx = old_prealloc_cnt1 * (partial_matching_len + 1) + partial_matching_len;
            d_new_head[idx] = v;
            for (int i = 0; i < partial_matching_len; i++) {
                idx = old_prealloc_cnt1 * (partial_matching_len + 1) + i;
                d_new_head[idx] = this_partial_matching[i];
            }
        }
    }
}

void __global__ Extend(
    const Graph_GPU &Q,
    const Graph_GPU &G,
    const candidate_graph_GPU &cg,
    int partial_matching_cnt,
    int partial_matching_len,
    int *d_head,
    int *d_new_head,
    int cur_query_vertex,
    int *d_rank
) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;

    int u = cur_query_vertex;
    int *this_partial_matching = d_head + partial_matching_len * warp_id;

    if (warp_id >= partial_matching_cnt) {
        return;
    }

    int fuu = Q.d_bknbrs_[Q.d_bknbrs_offset_[u]];
    assert(Q.d_bknbrs_offset_[u] <= Q.d_bknbrs_offset_[u + 1]);
    int fvv = this_partial_matching[d_rank[fuu]];
    int flen = 0;
    int *fset = cg.d_get_candidates(fuu, u, fvv, flen);

    for (int _i = lane_id; _i < flen; _i += warpSize) {
        int v = fset[_i];
        bool flag = true;
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
        for (int j = 0; j < partial_matching_len; j++) {
            if (this_partial_matching[j] == v) {
                flag = false;
            }
        }
        if (flag) {
            int old_prealloc_cnt1 = atomicAdd(&d_prealloc_cnt1, 1);
            int idx = old_prealloc_cnt1 * (partial_matching_len + 1) + partial_matching_len;
            d_new_head[idx] = v;
            for (int i = 0; i < partial_matching_len; i++) {
                idx = old_prealloc_cnt1 * (partial_matching_len + 1) + i;
                d_new_head[idx] = this_partial_matching[i];
            }
        }
    }
}

int join(
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

    int blocks = (partial_matching_cnt + warpsPerBlock - 1) / warpsPerBlock;
    // Extend<<<blocks, threadsPerBlock>>>(Q, G, cg, partial_matching_cnt, partial_matching_len,
    //                                     d_partial_matchings, d_new_partial_matchings, matching_order[partial_matching_len],
    //                                     d_rank);

    CHECK(cudaDeviceSynchronize());

    

    return 0;
}