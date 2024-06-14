#ifndef JOIN_H
#define JOIN_H

#include "candidate.h"
#include "graph_gpu.h"
#include "helper.h"

int *set_beginning_partial_matchings(
    const Graph &Q,
    const Graph &G,
    const candidate_graph &cg,
    const std::vector<int> &matching_order,
    int &cnt
);

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
);

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
);

int join(
    const Graph &q,
    const Graph &g,
    const Graph_GPU &Q, 
    const Graph_GPU &G, 
    const candidate_graph &_cg,
    const candidate_graph_GPU &cg,
    const std::vector<int> &matching_order
);

#endif