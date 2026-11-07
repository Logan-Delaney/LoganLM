#include <gtest/gtest.h>

#include <stdexcept>

#include "loganlm/storage.h"

TEST(Storage, ElementCount) {
    loganlm::Storage s;
    s.shape = {5, 2, 3};
    EXPECT_EQ(s.element_count(), 30u);
}

TEST(Storage, ScalarCount) {
    loganlm::Storage s;
    s.shape = {};
    EXPECT_EQ(s.element_count(), 1u);
}

TEST(Storage, OffsetOfMiddleElementUsesStride) {
    loganlm::Storage s;
    s.shape = {2, 3, 4};
    EXPECT_EQ(s.offset({1, 1, 2}), 18u);
}

TEST(Storage, OffsetOfFirstElementIsZero) {
    loganlm::Storage s;
    s.shape = {2, 3, 4};
    EXPECT_EQ(s.offset({0, 0, 0}), 0u);
}

TEST(Storage, OffsetOfLastElementIsCountMinusOne) {
    loganlm::Storage s;
    s.shape = {2, 3, 4};
    EXPECT_EQ(s.offset({1, 2, 3}), 23u);
}

TEST(Storage, OffsetThrowsOnWrongIndexCount) {
    loganlm::Storage s;
    s.shape = {2, 3};
    EXPECT_THROW((void)s.offset({0, 1, 2}), std::invalid_argument);
}

TEST(Storage, OffsetThrowsOnIndexOutOfRange) {
    loganlm::Storage s;
    s.shape = {2, 3, 4};
    EXPECT_THROW((void)s.offset({0, 3, 0}), std::out_of_range);
}

TEST(Storage, ReadingEveryIndexWalksDataInOrder) {
    loganlm::Storage s;
    s.shape = {2, 3, 4};
    for (std::size_t n = 0; n < s.element_count(); ++n) {
        s.data.push_back(static_cast<float>(n));
    }

    float expected = 0.0f;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                EXPECT_FLOAT_EQ(s.data[s.offset({i, j, k})], expected);
                expected += 1.0f;
            }
        }
    }
}

TEST(Storage, MemoryFootprintIsFourBytesPerElement) {
    loganlm::Storage s;
    s.shape = {5, 2, 3};
    s.data.resize(s.element_count());
    EXPECT_EQ(s.memory_footprint(), 120u);
}

TEST(Storage, MemoryFootprintOfScalarIsFourBytes) {
    loganlm::Storage s;
    s.data.resize(s.element_count());
    EXPECT_EQ(s.memory_footprint(), 4u);
}

TEST(Storage, MemoryFootprintMeasuresActualData) {
    loganlm::Storage s;
    s.shape = {5, 2, 3};
    EXPECT_EQ(s.memory_footprint(), 0u);
}

TEST(Storage, ShapeStringListsEveryDimension) {
    loganlm::Storage s;
    s.shape = {5, 2, 3};
    EXPECT_EQ(s.shape_string(), "(5, 2, 3)");
}

TEST(Storage, ShapeStringOfOneDimensionHasNoComma) {
    loganlm::Storage s;
    s.shape = {4};
    EXPECT_EQ(s.shape_string(), "(4)");
}

TEST(Storage, ShapeStringOfScalarIsEmptyParentheses) {
    loganlm::Storage s;
    EXPECT_EQ(s.shape_string(), "()");
}

TEST(Storage, ToStringOfScalarShowsItsValue) {
    loganlm::Storage s;
    s.data = {5.0f};
    EXPECT_EQ(s.to_string(), "shape=() 5");
}

TEST(Storage, ToStringOfVectorShowsFlatList) {
    loganlm::Storage s;
    s.shape = {3};
    s.data = {1.0f, 2.0f, 3.0f};
    EXPECT_EQ(s.to_string(), "shape=(3) [1, 2, 3]");
}

TEST(Storage, ToStringOfMatrixShowsOneGroupPerRow) {
    loganlm::Storage s;
    s.shape = {2, 3};
    s.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    EXPECT_EQ(s.to_string(), "shape=(2, 3) [[1, 2, 3], [4, 5, 6]]");
}

TEST(Storage, ToStringShowsValuesAtExactlyThirtyTwoElements) {
    loganlm::Storage s;
    s.shape = {32};
    s.data.assign(32, 0.0f);
    EXPECT_NE(s.to_string().find('['), std::string::npos);
}

TEST(Storage, ToStringOmitsValuesAboveThirtyTwoElements) {
    loganlm::Storage s;
    s.shape = {33};
    s.data.assign(33, 0.0f);
    EXPECT_EQ(s.to_string(), "shape=(33)");
}
