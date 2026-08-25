#pragma once

#include <cstdint>
#include <string>

struct LlamaWeightRequest {
    std::string model_dir;
    std::string layer_name;
    int block_index;
    int expected_m;
    int expected_k;
};

bool LoadLlamaWeight(
    const LlamaWeightRequest& request,
    uint16_t* destination,
    std::string* error_message
);