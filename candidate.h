#ifndef CANDIDATE_H
#define CANDIDATE_H

#include "graph.h"
#include "helper.h"

class candidate_graph {
    __host__ __inline__ int h_u_pair_key(int u1, int u2) const {
        return u1 * query_vertex_cnt + u2;
    }

public:
    std::vector<std::vector<int>> cand;
    int tot_cand_cnt;

    int query_vertex_cnt;
    int data_vertex_cnt;

    std::vector<int> h_cg_array;
    int *h_cg_offset;

    candidate_graph(Graph &Q, Graph &G);
};


class candidate_graph_GPU {
public:
    int *d_cand_set;
    int *d_cand_offset;

    int *d_query_vertex_cnt;
    int *d_data_vertex_cnt;

    int *d_cg_offset;
    int *d_cg_array;

public:
    candidate_graph_GPU(const candidate_graph &cg) {
        int V = cg.query_vertex_cnt;
        CHECK(cudaMalloc(&d_query_vertex_cnt, sizeof(int)));
        CHECK(cudaMemcpy(d_query_vertex_cnt, &V, sizeof(int), cudaMemcpyHostToDevice));
        int V_ = cg.data_vertex_cnt;
        CHECK(cudaMalloc(&d_data_vertex_cnt, sizeof(int)));
        CHECK(cudaMemcpy(d_data_vertex_cnt, &V_, sizeof(int), cudaMemcpyHostToDevice));

        int *h_cand_offset = (int *)malloc(sizeof(int) * (V + 1));
        h_cand_offset[0] = 0;
        h_cand_offset[V] = cg.tot_cand_cnt;
        for (int i = 1; i < V; i++) {
            h_cand_offset[i] = h_cand_offset[i - 1] + cg.cand[i - 1].size();
        }
        CHECK(cudaMalloc(&d_cand_offset, (V + 1) * sizeof(int)));
        CHECK(cudaMemcpy(d_cand_offset, h_cand_offset, (V + 1) * sizeof(int), cudaMemcpyHostToDevice));

        CHECK(cudaMalloc(&d_cand_set, cg.tot_cand_cnt * sizeof(int)));
        for (int i = 0; i < V; i++) {
            CHECK(cudaMemcpy(d_cand_set + h_cand_offset[i], cg.cand[i].data(),
                             sizeof(int) * (h_cand_offset[i + 1] - h_cand_offset[i]), cudaMemcpyHostToDevice));
        }
        printf("Candidate set moved to GPU.\n");

        int _offset_size = cg.query_vertex_cnt * cg.query_vertex_cnt * cg.data_vertex_cnt * sizeof(int);
        CHECK(cudaMalloc(&d_cg_array, sizeof(int) * cg.h_cg_array.size()));
        CHECK(cudaMemcpy(d_cg_array, cg.h_cg_array.data(), sizeof(int) * cg.h_cg_array.size(), cudaMemcpyHostToDevice));

        CHECK(cudaMalloc(&d_cg_offset, _offset_size));
        CHECK(cudaMemcpy(d_cg_offset, cg.h_cg_offset, _offset_size, cudaMemcpyHostToDevice));

        std::cout << "Total num of vertices in candidate_graph: " << cg.h_cg_array.size() << std::endl;
    }

    __device__ __inline__ int d_u_pair_key(int u1, int u2) const {
        return u1 * *d_query_vertex_cnt + u2;
    }

    __device__ int *d_get_candidates(int u1, int u2, int v, int &len) const {
        // when u1 is mapped to v, what are the candidates of u2 ?
        int *start = d_cg_offset + d_u_pair_key(u1, u2) * *d_data_vertex_cnt;
        len = start[v + 1] - start[v];
        assert(start[v] >= 0);
        assert(start[v + 1] >= 0);
        assert(len >= 0);
        return d_cg_array + start[v];
    }
};

#endif