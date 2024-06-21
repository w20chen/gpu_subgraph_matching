#ifndef HELPER_H
#define HELPER_H

#include <cstdio>
#include <string>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cstdint>
#include <assert.h>

#define ull unsigned long long

static const int Zero = 0;
static const int One = 1;

#define CHECK(call)                                   \
do {                                                  \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess) {                  \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)


struct pint {
    int first, second;

    pint() {
        first = second = 0;
    }

    pint(int a, int b) {
        first = a;
        second = b;
    }
};


static pint make_pint(int a, int b) {
    pint p(a, b);
    return p;
}


static void swap(uint32_t &a, uint32_t &b) {
    uint32_t t = a;
    a = b;
    b = t;
}


static void swap(int &a, int &b) {
    int t = a;
    a = b;
    b = t;
}


static void TODO() {
    printf("Implement me!\n");
    assert(0);
}


class InputParser {
public:
    InputParser(int &argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            tokens_.emplace_back(argv[i]);
        }
    }

    std::string get_cmd_option(const std::string &option) const {
        std::vector<std::string>::const_iterator itr;
        itr =  std::find(tokens_.begin(), tokens_.end(), option);
        if (itr != tokens_.end() && ++itr != tokens_.end()) {
            return *itr;
        }
        return "";
    }

    bool check_cmd_option_exists(const std::string &option) const {
        return std::find(tokens_.begin(), tokens_.end(), option)
               != tokens_.end();
    }

    std::string get_cmd() {
        return std::accumulate(tokens_.begin(), tokens_.end(), std::string(" "));
    }

private:
    std::vector<std::string> tokens_;
};


static __device__ bool 
binary_search(int *nums, int n, int value) {
    if (nums == nullptr) {
        return false;
    }
    int low = 0;
    int high = n - 1;
    while (low <= high) {
        int mid = low + ((high - low) >> 1);
        if (nums[mid] > value) {
            high = mid - 1;
        }
        else if (nums[mid] < value) {
            low = mid + 1;
        }
        else {
            return true;
        }
    }
    return false;
}


static void __global__ 
print_partial_results(int* head, int col, int row) {
    if (row > 20) {
        return;
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d ", *(head + col * i + j));
        }
        printf("\n");
    }
    printf("\n");
}

#endif