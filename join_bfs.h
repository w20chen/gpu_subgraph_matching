#ifndef JOIN_BFS_H
#define JOIN_BFS_H

#include "graph_gpu.h"
#include "candidate.h"
#include "mem_manager.h"


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