#ifndef JOIN_BFS_H
#define JOIN_BFS_H

#include "graph_gpu.h"
#include "candidate.h"
#include "join.h"


void __global__ BFS_Extend(
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

int join_bfs(
    const Graph &q,
    const Graph &g,
    const Graph_GPU &Q, 
    const Graph_GPU &G, 
    const candidate_graph &_cg,
    const candidate_graph_GPU &cg,
    const std::vector<int> &matching_order
);

#endif