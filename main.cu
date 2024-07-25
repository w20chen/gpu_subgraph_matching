#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include "helper.h"
#include "graph.h"
#include "graph_gpu.h"
#include "candidate.h"
#include "join_bfs.h"


int main(int argc, char **argv) {
    std::cout << sizeof(MemPool) << std::endl;
    InputParser cmd_parser(argc, argv);
    assert(cmd_parser.check_cmd_option_exists("-q"));
    assert(cmd_parser.check_cmd_option_exists("-d"));
    std::string input_query_graph_file = cmd_parser.get_cmd_option("-q");
    std::string input_data_graph_file = cmd_parser.get_cmd_option("-d");

    cudaSetDevice(0);
    check_gpu_props();

    Graph Q(input_query_graph_file, true);
    Graph G(input_data_graph_file, false);

    std::vector<int> matching_order;
    Q.generate_matching_order(matching_order);
    Q.generate_backward_neighborhood(matching_order);

    Graph_GPU Q_GPU(Q);
    Graph_GPU G_GPU(G);

    candidate_graph CG(Q, G);
    candidate_graph_GPU CG_GPU(CG);

    int ret = join_bfs(Q, G, Q_GPU, G_GPU, CG, CG_GPU, matching_order);
    printf("\033[41;37mResult: %d\033[0m\n", ret);

    Q_GPU.deallocate();
    G_GPU.deallocate();
    CG_GPU.deallocate();

    return 0;
}
