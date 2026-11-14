#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "loganlm/add.h"
#include "loganlm/storage.h"

TEST(Add, AddsElementWise) {
    loganlm::Storage a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    loganlm::Storage b({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, {2, 3});
    loganlm::Add add;

    loganlm::Storage result = add.forward({&a, &b});

    EXPECT_EQ(result.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(result.data(), (std::vector<float>{11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f}));
}

TEST(Add, AddsTwoScalars) {
    loganlm::Storage a({2.0f}, {});
    loganlm::Storage b({3.0f}, {});
    loganlm::Add add;

    loganlm::Storage result = add.forward({&a, &b});

    EXPECT_EQ(result.shape(), std::vector<std::size_t>{});
    EXPECT_EQ(result.data(), (std::vector<float>{5.0f}));
}

TEST(Add, LeavesInputsUnchanged) {
    loganlm::Storage a({1.0f, 2.0f, 3.0f}, {3});
    loganlm::Storage b({10.0f, 20.0f, 30.0f}, {3});
    loganlm::Add add;

    loganlm::Storage result = add.forward({&a, &b});

    EXPECT_EQ(a.data(), (std::vector<float>{1.0f, 2.0f, 3.0f}));
    EXPECT_EQ(b.data(), (std::vector<float>{10.0f, 20.0f, 30.0f}));
    EXPECT_NE(&result, &a);
    EXPECT_NE(&result, &b);
}

TEST(Add, ThrowsOnMismatchedShapesWithSameElementCount) {
    loganlm::Storage a(std::vector<float>(6, 0.0f), {2, 3});
    loganlm::Storage b(std::vector<float>(6, 0.0f), {3, 2});
    loganlm::Add add;

    EXPECT_THROW((void)add.forward({&a, &b}), std::invalid_argument);
}

TEST(Add, ThrowsOnMismatchedRank) {
    loganlm::Storage a(std::vector<float>(6, 0.0f), {2, 3});
    loganlm::Storage b(std::vector<float>(3, 0.0f), {3});
    loganlm::Add add;

    EXPECT_THROW((void)add.forward({&a, &b}), std::invalid_argument);
}

TEST(Add, ThrowsOnOneInput) {
    loganlm::Storage a({1.0f}, {});
    loganlm::Add add;

    EXPECT_THROW((void)add.forward({&a}), std::invalid_argument);
}

TEST(Add, ThrowsOnThreeInputs) {
    loganlm::Storage a({1.0f}, {});
    loganlm::Add add;

    EXPECT_THROW((void)add.forward({&a, &a, &a}), std::invalid_argument);
}

TEST(Add, AddsZeroSizedStoragesToZeroSizedResult) {
    loganlm::Storage a({}, {0, 3});
    loganlm::Storage b({}, {0, 3});
    loganlm::Add add;

    loganlm::Storage result = add.forward({&a, &b});

    EXPECT_EQ(result.shape(), (std::vector<std::size_t>{0, 3}));
    EXPECT_TRUE(result.data().empty());
}
