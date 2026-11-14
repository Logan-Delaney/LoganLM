#include "loganlm/tensor.h"

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
}  // namespace loganlm
