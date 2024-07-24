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
    // (1,82), (1,5460), ..., (37550,165), (37550,5734)
    // printf("(%d,%d), (%d,%d), ..., (%d,%d), (%d,%d)\n", 
    //     h_beginning_partial_matching.at(0).first, h_beginning_partial_matching.at(0).second,
    //     h_beginning_partial_matching.at(1).first, h_beginning_partial_matching.at(1).second,
    //     h_beginning_partial_matching.at(cnt - 2).first, h_beginning_partial_matching.at(cnt - 2).second,
    //     h_beginning_partial_matching.at(cnt - 1).first, h_beginning_partial_matching.at(cnt - 1).second);

    // FILE *fp = fopen("set_beg.txt", "w");
    // for (int i = 0; i < cnt; i++) {
    //     fprintf(fp, "%d,%d,\n", h_beginning_partial_matching[i].first, h_beginning_partial_matching[i].second);
    // }
    // fclose(fp);

    return d_dst;
}


void __global__ BFS_Extend(
    const Graph_GPU Q,
    const Graph_GPU G,
    const candidate_graph_GPU cg,
    MemManager *d_MM,
    int cur_query_vertex,
    int *d_rank,
    int partial_offset
) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int warp_id = tid / warpSize;
    int warp_id_in_blk = threadIdx.x / warpSize;
    int lane_id = tid % warpSize;

    // assign partial matching to each warp
    int u = cur_query_vertex;
    int partial_matching_len = 0;
    int *this_partial_matching = d_MM->get_partial(warp_id + partial_offset, &partial_matching_len);

    if (this_partial_matching == nullptr) {
        return;
    }

    assert(partial_matching_len >= 2);

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
        d_new_head = d_MM->write_mempool()->alloc();
        assert(d_new_head != nullptr);
        d_new_head_lower = (unsigned)d_new_head;
        // printf("%u %p\n", d_new_head_lower, d_new_head);
        d_new_head_upper = (unsigned)((unsigned long long)d_new_head >> 32);
    }
    d_new_head_lower = __shfl_sync(0xffffffff, d_new_head_lower, 0);
    d_new_head_upper = __shfl_sync(0xffffffff, d_new_head_upper, 0);
    d_new_head = (int *)(((unsigned long long)d_new_head_upper << 32) | (unsigned long long)d_new_head_lower);

    assert(d_new_head_upper != 0);
    assert(d_new_head != nullptr);
    // printf("%p\n", d_new_head);      // 0x7f19d3c00000

    // block writing counter for each warp
    extern __shared__ int blk_write_cnt[];
    blk_write_cnt[warp_id_in_blk] = 0;


    // compute extendable candidate set
    int ut = ceil_div(flen, warpSize);
    for (int t = 0; t < ut; t++) {
        int lid = t * warpSize + lane_id;
        // unsigned int loop_mask = 0;
        // printf("wid:%d, t:%d\n", warp_id, t);
        // __syncwarp();
        // loop_mask = __ballot_sync(0xffffffff, lid < flen);
        // printf("%d loop mask: %p\n", lid, loop_mask);
        // __syncwarp();
        if (lid >= flen) {
            break;
        }

        int v = fset[lid];

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

        // __syncwarp();

        // unsigned flag_mask = __ballot_sync(0xffffffff, flag);

        // v is good
        if (flag) {
            int old_cnt = atomicAdd(&blk_write_cnt[warp_id_in_blk], 1);

            // unsigned mask = __ballot_sync(flag_mask, (old_cnt + 1) * (partial_matching_len + 1) > d_MM->blockIntNum);
            // printf("mask: %p\n", mask);
            // allocate a new block for this warp
            if ((old_cnt + 1) * (partial_matching_len + 1) > d_MM->blockIntNum) {
                // printf("tid: %d, old_cnt: %d, partial_matching_len: %d\n", tid, old_cnt, partial_matching_len);
                // assert(0);
                blk_write_cnt[warp_id_in_blk] = 0;

                unsigned mask = __activemask();
                // printf("mask: %d\n", mask);  // 0xffffffff

                int leader = __ffs(mask) - 1;
                unsigned d_new_head_lower = 0;
                unsigned d_new_head_upper = 0;
                if (lane_id == leader) {
                    // add new props
                    partial_props p;
                    p.start_addr = d_new_head;
                    p.partial_len = partial_matching_len + 1;
                    p.partial_cnt = old_cnt;
                    d_MM->add_new_props(p);

                    d_new_head = d_MM->write_mempool()->alloc();
                    assert(d_new_head);
                    d_new_head_lower = (unsigned)d_new_head;
                    assert(d_new_head_lower);
                    d_new_head_upper = (unsigned)((unsigned long long)d_new_head >> 32);
                    assert(d_new_head_upper);
                    // printf("#%p %p\n", d_new_head_upper, d_new_head_lower);
                }
                __syncwarp();
                d_new_head_lower = __shfl_sync(0xffffffff, d_new_head_lower, leader);
                d_new_head_upper = __shfl_sync(0xffffffff, d_new_head_upper, leader);
                d_new_head = (int *)(((unsigned long long)d_new_head_upper << 32) | (unsigned long long)d_new_head_lower);
                __syncwarp();
                // assert(d_new_head_lower);
                // assert(d_new_head_upper);
                if (d_new_head == nullptr) {
                    printf("%p %p\n", d_new_head_upper, d_new_head_lower);
                }
                old_cnt = atomicAdd(&blk_write_cnt[warp_id_in_blk], 1);
                // __syncwarp(mask);
            }

            // __syncwarp(flag_mask);

            // write the newly found partial matching
            int idx = old_cnt * (partial_matching_len + 1) + partial_matching_len;
            assert(idx < d_MM->blockIntNum && idx >= 0);
            // printf("%d\n", idx);
            d_new_head[idx] = v;
            for (int i = 0; i < partial_matching_len; i++) {
                idx = old_cnt * (partial_matching_len + 1) + i;
                d_new_head[idx] = this_partial_matching[i];
            }
        }
        // __syncwarp();
    }

    __syncthreads();

    if (lane_id == 0) {
        if (blk_write_cnt[warp_id_in_blk] > 0) {
            partial_props p;
            p.start_addr = d_new_head;
            p.partial_len = partial_matching_len + 1;
            p.partial_cnt = blk_write_cnt[warp_id_in_blk];
            d_MM->add_new_props(p);
            // (0x7f9071c00000,3,2)
            // printf("(%p,%d,%d)\n", p.start_addr, p.partial_len, p.partial_cnt);
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

    MemManager h_MM;
    // Move a table of initial partial matchings (candidate edges) to memory pool.
    h_MM.init(d_partial_matchings, partial_matching_cnt);

    CHECK(cudaFree(d_partial_matchings));

    MemManager *d_MM = nullptr;
    CHECK(cudaMalloc(&d_MM, sizeof(MemManager)));

    // const int threadBlockNum = 1024;
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

        int partial_matching_cnt = h_MM.get_partial_cnt();
        printf("cnt: %d\n", partial_matching_cnt);

        CHECK(cudaMemcpy(d_MM, &h_MM, sizeof(MemManager), cudaMemcpyHostToDevice));

        // For each partial matching, assign a warp.
        int totalBlocks = ceil_div(partial_matching_cnt, warpsPerBlock);
        const int maxBlocks = 1;

        for (int partial_offset = 0; partial_offset < partial_matching_cnt; partial_offset += maxBlocks * warpsPerBlock) {
            // printf("call kernel <<<%d, %d, %d>>>\n", maxBlocks, threadsPerBlock, warpsPerBlock * sizeof(int));
            printf("partial_offset: %d / %d\n", partial_offset, partial_matching_cnt);
            BFS_Extend<<<maxBlocks, threadsPerBlock, warpsPerBlock * sizeof(int)>>>(
                Q, G, cg, d_MM, matching_order[partial_matching_len], d_rank, partial_offset
            );
            CHECK(cudaDeviceSynchronize());
        }

        CHECK(cudaMemcpy(&h_MM, d_MM, sizeof(MemManager), cudaMemcpyDeviceToHost));
        h_MM.swap_mem_pool();
        CHECK(cudaDeviceSynchronize());
    }

    int ret = h_MM.get_partial_cnt();

    // h_MM.dump("result.txt");

    return ret;
}