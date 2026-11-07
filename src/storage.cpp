#include "loganlm/storage.h"

#include <sstream>
#include <stdexcept>

namespace loganlm {

std::size_t Storage::element_count() const {
    std::size_t total = 1;
    for (const std::size_t d : shape) {
        total *= d;
    }
    return total;
}

std::size_t Storage::offset(const std::vector<std::size_t>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("Invalid indices size");
    }
    std::size_t result = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of range");
        }
        std::size_t stride = 1;
        for (std::size_t j = i + 1; j < shape.size(); ++j) {
            stride *= shape[j];
        }
        result += stride * indices[i];
    }
    return result;
}

std::size_t Storage::memory_footprint() const {
    return data.size() * sizeof(float);
}

std::string Storage::shape_string() const {
    std::string text = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            text += ", ";
        }
        text += std::to_string(shape[i]);
    }
    text += ')';
    return text;
}

namespace {

constexpr std::size_t max_printed_elements = 32;

void write_slice(std::ostringstream& out, const std::vector<float>& data,
                 const std::vector<std::size_t>& shape, std::size_t dim, std::size_t start) {
    std::size_t stride = 1;
    for (std::size_t j = dim + 1; j < shape.size(); ++j) {
        stride *= shape[j];
    }

    out << '[';
    for (std::size_t i = 0; i < shape[dim]; ++i) {
        if (i != 0) {
            out << ", ";
        }
        if (dim + 1 == shape.size()) {
            out << data[start + i];
        } else {
            write_slice(out, data, shape, dim + 1, start + i * stride);
        }
    }
    out << ']';
}

}  // namespace

std::string Storage::to_string() const {
    std::string text = "shape=" + shape_string();
    if (element_count() > max_printed_elements) {
        return text;
    }

    std::ostringstream out;
    if (shape.empty()) {
        out << data[0];
    } else {
        write_slice(out, data, shape, 0, 0);
    }
    return text + ' ' + out.str();
}

}  // namespace loganlm
