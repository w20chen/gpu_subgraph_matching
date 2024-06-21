#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <assert.h>
#include <fstream>
#include <string>
#include <algorithm>


class Graph {
private:
    bool is_query_;

    int vcount_;
    int ecount_;

    std::vector<int> deg_;
    std::vector<int> vertex_label_;

    std::vector<int> bknbrs_;
    std::vector<int> bknbrs_offset_;

public:

    friend class Graph_GPU;

    std::vector<std::vector<int>> adj_;
    std::vector<std::unordered_map<int, int>> nlf_;
    std::unordered_map<int, std::vector<int>> label_vertex_mapping_;

    bool is_query() const {
        return is_query_;
    }

    int max_label() const {
        return *std::max_element(vertex_label_.begin(), vertex_label_.end());
    }

    int label(int u) const {
        return vertex_label_[u];
    }

    int vcount() const {
        return vcount_;
    }

    int ecount() const {
        return ecount_;
    }

    Graph(const std::string &file_path, bool is_q) {
        is_query_ = is_q;

        std::ifstream infile(file_path);
        if (!infile.is_open()) {
            std::cout << "Cannot open graph file " << file_path << "." << std::endl;
            exit(-1);
        }

        char type = 0;
        infile >> type >> vcount_ >> ecount_;
        assert(type == 't');

        adj_.resize(vcount_);
        deg_.resize(vcount_);
        vertex_label_.resize(vcount_);
        nlf_.resize(vcount_);

        while (infile >> type) {
            if (type == 'v') {
                int vid, label, deg;
                infile >> vid >> label >> deg;
                vertex_label_[vid] = label;
                deg_[vid] = deg;
                label_vertex_mapping_[label].emplace_back(vid);
            }
            else {
                assert(type == 'e');
                break;
            }
        }

        if (type == 'e') {
            do {
                int v1, v2;
                infile >> v1 >> v2;

                adj_[v1].emplace_back(v2);
                adj_[v2].emplace_back(v1);

                nlf_[v1][vertex_label_[v2]]++;
                nlf_[v2][vertex_label_[v1]]++;
            }
            while (infile >> type);
        }

        infile.close();

        for (int v = 0; v < vcount_; v++) {
            assert(adj_[v].size() == deg_[v]);
        }

        for (auto &l : adj_) {
            std::sort(l.begin(), l.end());
        }

        std::cout << "Graph loaded from file " << file_path << "." << std::endl;

        print_meta();
    }

    void print_meta() const {
        std::cout << "|V|=" << vcount_ << " |E|=" << ecount_ << " |L|=" << label_vertex_mapping_.size() << std::endl;
    }

    bool is_adjacent(int v1, int v2) const {
        if (adj_[v1].size() < adj_[v2].size()) {
            auto it = std::lower_bound(adj_[v1].begin(), adj_[v1].end(), v2);
            if (it == adj_[v1].end()) {
                return false;
            }
            else {
                return *it == v2;
            }
        }

        auto it = std::lower_bound(adj_[v2].begin(), adj_[v2].end(), v1);
        if (it == adj_[v2].end()) {
            return false;
        }
        else {
            return *it == v1;
        }
    }

    void generate_backward_neighborhood(const std::vector<int> &matching_order) {
        assert(matching_order.size() == vcount_);
        // Only query graph can call this function.
        assert(is_query_);

        bknbrs_offset_.resize(vcount_ + 1);
        for (int u = 0; u < vcount_; u++) {
            bknbrs_offset_[u] = bknbrs_.size();
            for (int i = 0; i < matching_order.size(); i++) {
                int uu = (int)matching_order[i];
                if (uu == u) {
                    break;
                }
                else if (is_adjacent(u, uu)) {
                    bknbrs_.push_back(uu);
                }
            }
        }
        bknbrs_offset_[vcount_] = bknbrs_.size();

        assert(bknbrs_offset_.size() != 0);
        assert(bknbrs_.size() != 0);
    }

    void generate_matching_order(std::vector<int> &matching_order) const;
};



#endif