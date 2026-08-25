#include "llama_weight_loader.h"

#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

#include <nlohmann/json.hpp>

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

bool Fail(
    std::string* error_message,
    const std::string& message)
{
    if (error_message != nullptr) {
        *error_message = message;
    }
    return false;
}

std::string JoinPath(
    const std::string& directory,
    const std::string& filename)
{
    if (directory.empty()) {
        return filename;
    }

    if (directory.back() == '/') {
        return directory + filename;
    }

    return directory + "/" + filename;
}

bool LoadIndex(
    const std::string& model_dir,
    Json* index,
    std::string* error_message)
{
    const std::string index_path =
        JoinPath(model_dir, "model.safetensors.index.json");

    std::ifstream input(index_path);

    if (!input.is_open()) {
        return Fail(
            error_message,
            "Cannot open safetensors index: " + index_path
        );
    }

    try {
        input >> *index;
    } catch (const std::exception& exception) {
        return Fail(
            error_message,
            "Cannot parse safetensors index " +
            index_path + ": " + exception.what()
        );
    }

    if (!index->contains("weight_map") ||
        !index->at("weight_map").is_object()) {
        return Fail(
            error_message,
            "Invalid safetensors index: weight_map is missing"
        );
    }

    return true;
}

bool FindShard(
    const Json& index,
    const std::string& tensor_name,
    std::string* shard_name,
    std::string* error_message)
{
    const Json& weight_map = index.at("weight_map");
    const auto iterator = weight_map.find(tensor_name);

    if (iterator == weight_map.end() ||
        !iterator->is_string()) {
        return Fail(
            error_message,
            "Tensor is missing from weight_map: " + tensor_name
        );
    }

    *shard_name = iterator->get<std::string>();
    return true;
}

bool CopyTensorFromShard(
    const std::string& shard_path,
    const std::string& tensor_name,
    size_t expected_k,
    size_t maximum_rows,
    uint16_t* destination,
    size_t* copied_rows,
    std::string* error_message)
{
    safetensors::safetensors_t safetensor;
    std::string warning;
    std::string loader_error;

    // shard 전체를 RAM에 복사하지 않고 mmap한다.
    const bool loaded = safetensors::mmap_from_file(
        shard_path,
        &safetensor,
        &warning,
        &loader_error
    );

    if (!warning.empty()) {
        std::cerr
            << "Safetensors warning: "
            << warning << "\n";
    }

    if (!loaded) {
        return Fail(
            error_message,
            "Cannot mmap shard " + shard_path +
            ": " + loader_error
        );
    }

    std::string validation_error;

    if (!safetensors::validate_data_offsets(
            safetensor,
            validation_error)) {
        return Fail(
            error_message,
            "Invalid tensor offsets in " + shard_path +
            ": " + validation_error
        );
    }

    safetensors::tensor_t tensor;

    if (!safetensor.tensors.at(tensor_name, &tensor)) {
        return Fail(
            error_message,
            "Tensor not found in shard " +
            shard_path + ": " + tensor_name
        );
    }

    if (tensor.dtype != safetensors::dtype::kBFLOAT16) {
        return Fail(
            error_message,
            "Tensor is not BF16: " + tensor_name +
            ", dtype=" +
            safetensors::get_dtype_str(tensor.dtype)
        );
    }

    if (tensor.shape.size() != 2) {
        return Fail(
            error_message,
            "Tensor must be two-dimensional: " +
            tensor_name
        );
    }

    const size_t rows = tensor.shape[0];
    const size_t cols = tensor.shape[1];

    if (cols != expected_k) {
        std::ostringstream message;
        message
            << tensor_name
            << ": expected K=" << expected_k
            << ", actual shape=["
            << rows << "," << cols << "]";

        return Fail(error_message, message.str());
    }

    if (rows > maximum_rows) {
        std::ostringstream message;
        message
            << tensor_name
            << ": tensor rows exceed destination; rows="
            << rows
            << ", remaining=" << maximum_rows;

        return Fail(error_message, message.str());
    }

    const size_t expected_bytes =
        rows * cols * sizeof(uint16_t);

    const size_t begin = tensor.data_offsets[0];
    const size_t end = tensor.data_offsets[1];

    if (end < begin || end - begin != expected_bytes) {
        std::ostringstream message;
        message
            << tensor_name
            << ": invalid data size; expected="
            << expected_bytes
            << ", actual="
            << (end >= begin ? end - begin : 0);

        return Fail(error_message, message.str());
    }

    if (!safetensor.mmaped ||
        safetensor.databuffer_addr == nullptr) {
        return Fail(
            error_message,
            "Safetensor data buffer unavailable: " +
            shard_path
        );
    }

    // data_offsets는 databuffer_addr 기준 상대 위치이다.
    const uint8_t* source =
        safetensor.databuffer_addr + begin;

    std::memcpy(
        destination,
        source,
        expected_bytes
    );

    *copied_rows = rows;

    std::cout
        << "Loaded tensor: " << tensor_name
        << ", shape=[" << rows << "," << cols << "]"
        << ", shard=" << shard_path
        << "\n";

    return true;
}

bool GetSourceTensorNames(
    const LlamaWeightRequest& request,
    std::vector<std::string>* tensor_names,
    std::string* error_message)
{
    tensor_names->clear();

    if (request.layer_name == "lm_head") {
        tensor_names->push_back("lm_head.weight");
        return true;
    }

    if (request.block_index < 0) {
        return Fail(
            error_message,
            "block_index must be non-negative"
        );
    }

    const std::string prefix =
        "model.layers." +
        std::to_string(request.block_index);

    if (request.layer_name == "qkv_proj") {
        tensor_names->push_back(
            prefix + ".self_attn.q_proj.weight"
        );
        tensor_names->push_back(
            prefix + ".self_attn.k_proj.weight"
        );
        tensor_names->push_back(
            prefix + ".self_attn.v_proj.weight"
        );
    } else if (request.layer_name == "o_proj") {
        tensor_names->push_back(
            prefix + ".self_attn.o_proj.weight"
        );
    } else if (request.layer_name == "gateup_proj") {
        tensor_names->push_back(
            prefix + ".mlp.gate_proj.weight"
        );
        tensor_names->push_back(
            prefix + ".mlp.up_proj.weight"
        );
    } else if (request.layer_name == "down_proj") {
        tensor_names->push_back(
            prefix + ".mlp.down_proj.weight"
        );
    } else {
        return Fail(
            error_message,
            "Unsupported Llama layer: " +
            request.layer_name
        );
    }

    return true;
}

}  // namespace

bool LoadLlamaWeight(
    const LlamaWeightRequest& request,
    uint16_t* destination,
    std::string* error_message)
{
    if (destination == nullptr) {
        return Fail(
            error_message,
            "Weight destination is null"
        );
    }

    if (request.expected_m <= 0 ||
        request.expected_k <= 0) {
        return Fail(
            error_message,
            "Expected M and K must be positive"
        );
    }

    Json index;

    if (!LoadIndex(
            request.model_dir,
            &index,
            error_message)) {
        return false;
    }

    std::vector<std::string> tensor_names;

    if (!GetSourceTensorNames(
            request,
            &tensor_names,
            error_message)) {
        return false;
    }

    const size_t expected_m =
        static_cast<size_t>(request.expected_m);
    const size_t expected_k =
        static_cast<size_t>(request.expected_k);

    size_t total_rows = 0;

    for (const std::string& tensor_name : tensor_names) {
        std::string shard_name;

        if (!FindShard(
                index,
                tensor_name,
                &shard_name,
                error_message)) {
            return false;
        }

        const std::string shard_path =
            JoinPath(request.model_dir, shard_name);

        size_t tensor_rows = 0;

        if (!CopyTensorFromShard(
                shard_path,
                tensor_name,
                expected_k,
                expected_m - total_rows,
                destination + total_rows * expected_k,
                &tensor_rows,
                error_message)) {
            return false;
        }

        total_rows += tensor_rows;
    }

    if (total_rows != expected_m) {
        std::ostringstream message;
        message
            << request.layer_name
            << ": combined shape mismatch; expected M="
            << expected_m
            << ", actual M=" << total_rows;

        return Fail(error_message, message.str());
    }

    std::cout
        << "Loaded Llama projection: "
        << request.layer_name
        << ", block=" << request.block_index
        << ", shape=[" << expected_m
        << "," << expected_k << "]\n";

    return true;
}