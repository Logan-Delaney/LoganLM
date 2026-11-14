#pragma once

#include <stdexcept>
#include <vector>

#include "loganlm/storage.h"

namespace loganlm {
class Operation {
public:
    virtual ~Operation() = default;

    [[nodiscard]] virtual Storage forward(const std::vector<const Storage*>& inputs) const = 0;

    [[nodiscard]] virtual std::vector<Storage> backward(const Storage& grad_output) const;

private:
};
}  // namespace loganlm
