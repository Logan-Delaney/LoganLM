#pragma once
#include "loganlm/operation.h"

namespace loganlm {
class Add : public Operation {
public:
    [[nodiscard]] Storage forward(const std::vector<const Storage*>& inputs) const override;
};
}  // namespace loganlm
