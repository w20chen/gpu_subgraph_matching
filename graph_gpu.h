#ifndef GRAPH_GPU_H
#define GRAPH_GPU_H

#include "graph.h"
#include "helper.h"

class Graph_GPU {
public:
    int *d_adj_;
    int *d_offset_;

    int *d_bknbrs_;
    int *d_bknbrs_offset_;

public:
    Graph_GPU(const Graph& G) {
        int *h_adj_ = (int *)malloc(sizeof(int) * G.ecount() * 2);
        int *h_offset_ = (int *)malloc(sizeof(int) * (G.vcount() + 1));
        int off = 0;
        for (int i = 0; i < G.vcount(); i++) {
            h_offset_[i] = off;
            for (int j : G.adj_[i]) {
                h_adj_[off++] = j;
            }
        }
        h_offset_[G.vcount()] = G.ecount() * 2;
        assert(off == G.ecount() * 2);

        CHECK(cudaMalloc(&d_adj_, sizeof(int) * G.ecount() * 2));
        CHECK(cudaMalloc(&d_offset_, sizeof(int) * (G.vcount() + 1)));
        CHECK(cudaMemcpy(d_adj_, h_adj_, sizeof(int) * G.ecount() * 2, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_offset_, h_offset_, sizeof(int) * (G.vcount() + 1), cudaMemcpyHostToDevice));

        free(h_adj_);
        free(h_offset_);

        d_bknbrs_ = nullptr;
        d_bknbrs_offset_ = nullptr;
        if (G.is_query()) {
            assert(G.bknbrs_.size() != 0);
            assert(G.bknbrs_offset_.size() != 0 && G.bknbrs_offset_.size() == G.vcount() + 1);
            CHECK(cudaMalloc(&d_bknbrs_, sizeof(int) * G.bknbrs_.size()));
            CHECK(cudaMalloc(&d_bknbrs_offset_, sizeof(int) * G.bknbrs_offset_.size()));
            CHECK(cudaMemcpy(d_bknbrs_, G.bknbrs_.data(), sizeof(int) * G.bknbrs_.size(), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_bknbrs_offset_, G.bknbrs_offset_.data(), sizeof(int) * G.bknbrs_offset_.size(), cudaMemcpyHostToDevice));
        }
    }

    void deallocate() {
        CHECK(cudaFree(d_adj_));
        CHECK(cudaFree(d_offset_));
        if (d_bknbrs_ != nullptr) {
            CHECK(cudaFree(d_bknbrs_));
        }
        if (d_bknbrs_offset_ != nullptr) {
            CHECK(cudaFree(d_bknbrs_offset_));
        }
    }
};

#endif