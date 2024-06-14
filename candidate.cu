#include "graph.h"
#include "candidate.h"
#include <vector>
#include <unordered_map>

candidate_graph::candidate_graph(Graph &Q, Graph &G) {
    tot_cand_cnt = 0;
    query_vertex_cnt = Q.vcount();
    data_vertex_cnt = G.vcount();
    cand.resize(Q.vcount());

    for (int u = 0; u < Q.vcount(); u++) {
        int label = Q.label(u);
        for (int v : G.label_vertex_mapping_[label]) {
            bool fail = false;
            for (auto p : Q.nlf_[u]) {
                int label = p.first;
                int count = p.second;
                if (count > G.nlf_[v][label]) {
                    fail = true;
                    break;
                }
            }
            if (!fail) {
                cand[u].emplace_back(v);
            }
        }
    }

    tot_cand_cnt = 0;
    for (int i = 0; i < cand.size(); i++) {
        tot_cand_cnt += cand[i].size();
        std::sort(cand[i].begin(), cand[i].end());
        std::cout << "Candidate num of " << i << ": " << cand[i].size() << std::endl;
    }
    std::cout << "Total num of candidates: " << tot_cand_cnt << std::endl;

    // Build candidate graph on the CPU.
    int _offset_size = Q.vcount() * Q.vcount() * G.vcount() * sizeof(int);
    h_cg_offset = (int *)malloc(_offset_size);
    memset(h_cg_offset, -1, _offset_size);

    for (int u1 = 0; u1 < Q.vcount(); u1++) {
        for (int u2 = 0; u2 < Q.vcount(); u2++) {
            if (Q.is_adjacent(u1, u2)) {
                int row = h_u_pair_key(u1, u2);
                int *start = h_cg_offset + row * G.vcount();
                for (int v = 0; v < G.vcount(); v++) {
                    // when u1 is mapped to v, what are the candidates of u2 ?
                    start[v] = h_cg_array.size();
                    // intersect C(u2) and N(v)
                    for (int vv : cand[u2]) {
                        if (G.is_adjacent(v, vv)) {
                            h_cg_array.push_back(vv);
                        }
                    }
                    std::sort(h_cg_array.begin() + start[v], h_cg_array.end());
                    start[v + 1] = h_cg_array.size();
                }
            }
        }
    }
}