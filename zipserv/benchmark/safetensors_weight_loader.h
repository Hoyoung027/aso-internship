#pragma once

#include <cstdint>
#include <string>
#include <vector>

// The experiment runner resolves model-specific tensor names from the JSON
// configuration.  The C++ loader therefore only needs to understand the
// safetensors container and how to concatenate 2-D BF16 tensors by rows.
struct SafetensorsWeightRequest {
    std::string model_dir;
    std::string label;
    std::vector<std::string> tensor_names;
    int expected_m;
    int expected_k;
};

bool LoadSafetensorsWeight(
    const SafetensorsWeightRequest& request,
    uint16_t* destination,
    std::string* error_message
);
