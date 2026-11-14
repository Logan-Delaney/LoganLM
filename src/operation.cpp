#include "loganlm/operation.h"

namespace loganlm {
std::vector<Storage> Operation::backward(const Storage&) const {
    throw std::logic_error("Operation::backward is not implemented");
}
}  // namespace loganlm
