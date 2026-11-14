#pragma once
#include <vector>

#include "loganlm/operation.h"
#include "loganlm/storage.h"

namespace loganlm {
class Tensor {
public:
    explicit Tensor(Storage storage);
    [[nodiscard]] const Storage& storage() const;

private:
    Storage storage_;
};
[[nodiscard]] Tensor apply(const Operation& op, const std::vector<const Tensor*>& inputs);
[[nodiscard]] Tensor stack(const std::vector<const Tensor*>& tensors);
}  // namespace loganlm
