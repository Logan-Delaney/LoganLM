#include "loganlm/storage.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace loganlm {
Storage::Storage(std::vector<float> data, std::vector<std::size_t> shape)
    : data_(std::move(data)), shape_(std::move(shape)) {
    if (data_.size() != element_count()) {
        throw std::invalid_argument("Data does not match shape");
    }
}

std::size_t Storage::element_count() const {
    std::size_t total = 1;
    for (const std::size_t d : shape_) {
        total *= d;
    }
    return total;
}

std::size_t Storage::offset(const std::vector<std::size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Invalid indices size");
    }
    std::size_t result = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of range");
        }
        std::size_t stride = 1;
        for (std::size_t j = i + 1; j < shape_.size(); ++j) {
            stride *= shape_[j];
        }
        result += stride * indices[i];
    }
    return result;
}

std::size_t Storage::memory_footprint() const {
    return data_.size() * sizeof(float);
}

std::string Storage::shape_string() const {
    std::string text = "(";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (i != 0) {
            text += ", ";
        }
        text += std::to_string(shape_[i]);
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
    if (shape_.empty()) {
        out << data_[0];
    } else {
        write_slice(out, data_, shape_, 0, 0);
    }
    return text + ' ' + out.str();
}

const std::vector<float>& Storage::data() const {
    return data_;
}

const std::vector<std::size_t>& Storage::shape() const {
    return shape_;
}
}  // namespace loganlm
