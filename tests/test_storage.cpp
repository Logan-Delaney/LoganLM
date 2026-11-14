#include <gtest/gtest.h>

#include <stdexcept>

#include "loganlm/storage.h"

namespace {
std::vector<float> zeros(std::size_t count) {
    return std::vector<float>(count, 0.0f);
}
}  // namespace

TEST(StorageConstruction, AcceptsDataThatMatchesShape) {
    loganlm::Storage s({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    EXPECT_EQ(s.element_count(), 6u);
}

TEST(StorageConstruction, ThrowsWhenDataIsTooShort) {
    EXPECT_THROW((void)loganlm::Storage({1.0f, 2.0f}, {2, 3}), std::invalid_argument);
}

TEST(StorageConstruction, ThrowsWhenDataIsTooLong) {
    EXPECT_THROW((void)loganlm::Storage(zeros(7), {2, 3}), std::invalid_argument);
}

TEST(StorageConstruction, AcceptsScalarWithOneValue) {
    loganlm::Storage s({5.0f}, {});
    EXPECT_EQ(s.element_count(), 1u);
}

TEST(StorageConstruction, ThrowsOnScalarWithNoData) {
    EXPECT_THROW((void)loganlm::Storage({}, {}), std::invalid_argument);
}

TEST(StorageConstruction, AcceptsZeroSizedDimensionWithNoData) {
    loganlm::Storage s({}, {0, 3});
    EXPECT_EQ(s.element_count(), 0u);
}

TEST(StorageConstruction, ThrowsOnZeroSizedDimensionWithData) {
    EXPECT_THROW((void)loganlm::Storage(zeros(3), {0, 3}), std::invalid_argument);
}

TEST(StorageConstruction, KeepsTheValuesItWasGiven) {
    loganlm::Storage s({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_EQ(s.data(), (std::vector<float>{1.0f, 2.0f, 3.0f}));
    EXPECT_EQ(s.shape(), (std::vector<std::size_t>{3}));
}

TEST(Storage, ElementCount) {
    loganlm::Storage s(zeros(30), {5, 2, 3});
    EXPECT_EQ(s.element_count(), 30u);
}

TEST(Storage, ScalarCount) {
    loganlm::Storage s(zeros(1), {});
    EXPECT_EQ(s.element_count(), 1u);
}

TEST(Storage, OffsetOfMiddleElementUsesStride) {
    loganlm::Storage s(zeros(24), {2, 3, 4});
    EXPECT_EQ(s.offset({1, 1, 2}), 18u);
}

TEST(Storage, OffsetOfFirstElementIsZero) {
    loganlm::Storage s(zeros(24), {2, 3, 4});
    EXPECT_EQ(s.offset({0, 0, 0}), 0u);
}

TEST(Storage, OffsetOfLastElementIsCountMinusOne) {
    loganlm::Storage s(zeros(24), {2, 3, 4});
    EXPECT_EQ(s.offset({1, 2, 3}), 23u);
}

TEST(Storage, OffsetThrowsOnWrongIndexCount) {
    loganlm::Storage s(zeros(6), {2, 3});
    EXPECT_THROW((void)s.offset({0, 1, 2}), std::invalid_argument);
}

TEST(Storage, OffsetThrowsOnIndexOutOfRange) {
    loganlm::Storage s(zeros(24), {2, 3, 4});
    EXPECT_THROW((void)s.offset({0, 3, 0}), std::out_of_range);
}

TEST(Storage, OffsetThrowsOnAnyIndexIntoZeroSizedDimension) {
    loganlm::Storage s({}, {0, 3});
    EXPECT_THROW((void)s.offset({0, 0}), std::out_of_range);
}

TEST(Storage, ReadingEveryIndexWalksDataInOrder) {
    std::vector<float> values;
    for (std::size_t n = 0; n < 24; ++n) {
        values.push_back(static_cast<float>(n));
    }
    loganlm::Storage s(std::move(values), {2, 3, 4});

    float expected = 0.0f;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                EXPECT_FLOAT_EQ(s.data()[s.offset({i, j, k})], expected);
                expected += 1.0f;
            }
        }
    }
}

TEST(Storage, MemoryFootprintIsFourBytesPerElement) {
    loganlm::Storage s(zeros(30), {5, 2, 3});
    EXPECT_EQ(s.memory_footprint(), 120u);
}

TEST(Storage, MemoryFootprintOfScalarIsFourBytes) {
    loganlm::Storage s(zeros(1), {});
    EXPECT_EQ(s.memory_footprint(), 4u);
}

TEST(Storage, MemoryFootprintOfZeroSizedStorageIsZero) {
    loganlm::Storage s({}, {0, 3});
    EXPECT_EQ(s.memory_footprint(), 0u);
}

TEST(Storage, ShapeStringListsEveryDimension) {
    loganlm::Storage s(zeros(30), {5, 2, 3});
    EXPECT_EQ(s.shape_string(), "(5, 2, 3)");
}

TEST(Storage, ShapeStringOfOneDimensionHasNoComma) {
    loganlm::Storage s(zeros(4), {4});
    EXPECT_EQ(s.shape_string(), "(4)");
}

TEST(Storage, ShapeStringOfScalarIsEmptyParentheses) {
    loganlm::Storage s(zeros(1), {});
    EXPECT_EQ(s.shape_string(), "()");
}

TEST(Storage, ToStringOfScalarShowsItsValue) {
    loganlm::Storage s({5.0f}, {});
    EXPECT_EQ(s.to_string(), "shape=() 5");
}

TEST(Storage, ToStringOfVectorShowsFlatList) {
    loganlm::Storage s({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_EQ(s.to_string(), "shape=(3) [1, 2, 3]");
}

TEST(Storage, ToStringOfMatrixShowsOneGroupPerRow) {
    loganlm::Storage s({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    EXPECT_EQ(s.to_string(), "shape=(2, 3) [[1, 2, 3], [4, 5, 6]]");
}

TEST(Storage, ToStringOfZeroSizedStorageShowsEmptyList) {
    loganlm::Storage s({}, {0, 3});
    EXPECT_EQ(s.to_string(), "shape=(0, 3) []");
}

TEST(Storage, ToStringShowsValuesAtExactlyThirtyTwoElements) {
    loganlm::Storage s(zeros(32), {32});
    EXPECT_NE(s.to_string().find('['), std::string::npos);
}

TEST(Storage, ToStringOmitsValuesAboveThirtyTwoElements) {
    loganlm::Storage s(zeros(33), {33});
    EXPECT_EQ(s.to_string(), "shape=(33)");
}
