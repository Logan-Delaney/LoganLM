#include "loganlm/tensor.h"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace loganlm {
Tensor::Tensor(Storage storage) : storage_(std::move(storage)) {
}

const Storage& Tensor::storage() const {
    return storage_;
}

Tensor apply(const Operation& op, const std::vector<const Tensor*>& inputs) {
    std::vector<const Storage*> input_storages;
    input_storages.reserve(inputs.size());
    for (const Tensor* input : inputs) {
        input_storages.push_back(&input->storage());
    }
    return Tensor{op.forward(input_storages)};
}

Tensor stack(const std::vector<const Tensor*>& tensors) {
    if (tensors.empty()) {
        throw std::invalid_argument("Tensor list cannot be empty");
    }
    const std::vector<std::size_t>& base_shape = tensors[0]->storage().shape();
    for (const Tensor* tensor : tensors) {
        if (tensor->storage().shape() != base_shape) {
            throw std::invalid_argument("Shape of all tensors must match");
        }
    }
    std::vector<std::size_t> stacked_shape;
    stacked_shape.reserve(base_shape.size() + 1);
    stacked_shape.push_back(tensors.size());
    stacked_shape.insert(stacked_shape.end(), base_shape.begin(), base_shape.end());
    std::vector<float> stacked_data;
    stacked_data.reserve(tensors[0]->storage().data().size() * tensors.size());
    for (const Tensor* tensor : tensors) {
        const std::vector<float>& data = tensor->storage().data();
        stacked_data.insert(stacked_data.end(), data.begin(), data.end());
    }
    return Tensor{Storage{std::move(stacked_data), std::move(stacked_shape)}};
}
}  // namespace loganlm
