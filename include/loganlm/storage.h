#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace loganlm {
class Storage {
public:
    Storage(std::vector<float> data, std::vector<std::size_t> shape);

    [[nodiscard]] std::size_t element_count() const;

    [[nodiscard]] std::size_t offset(const std::vector<std::size_t>& indices) const;

    [[nodiscard]] std::size_t memory_footprint() const;

    [[nodiscard]] std::string shape_string() const;

    [[nodiscard]] std::string to_string() const;

    [[nodiscard]] const std::vector<float>& data() const;

    [[nodiscard]] const std::vector<std::size_t>& shape() const;

private:
    std::vector<float> data_;
    std::vector<std::size_t> shape_;
};
}  // namespace loganlm
