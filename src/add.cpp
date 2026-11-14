#include "loganlm/add.h"

#include <stdexcept>
#include <utility>

namespace loganlm {
Storage Add::forward(const std::vector<const Storage*>& inputs) const {
    if (inputs.size() != 2) {
        throw std::invalid_argument("Number of inputs must be 2");
    }
    const Storage& a = *inputs[0];
    const Storage& b = *inputs[1];
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Input shapes must be identical");
    }
    std::vector<float> sums(a.data().size());
    for (std::size_t i = 0; i < a.data().size(); ++i) {
        sums[i] = a.data()[i] + b.data()[i];
    }
    return {std::move(sums), a.shape()};
}
}  // namespace loganlm
