#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "loganlm/add.h"
#include "loganlm/operation.h"
#include "loganlm/storage.h"
#include "loganlm/tensor.h"

TEST(Tensor, KeepsTheStorageItWasGiven) {
    loganlm::Tensor t(loganlm::Storage({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}));

    EXPECT_EQ(t.storage().shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(t.storage().data(), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

TEST(Tensor, CopyOwnsItsOwnNumbers) {
    loganlm::Tensor original(loganlm::Storage({1.0f, 2.0f, 3.0f}, {3}));

    loganlm::Tensor copy = original;

    EXPECT_EQ(copy.storage().data(), original.storage().data());
    EXPECT_NE(copy.storage().data().data(), original.storage().data().data());
}

TEST(Tensor, ConstructingFromMovedStorageDoesNotCopyTheNumbers) {
    loganlm::Storage storage({1.0f, 2.0f, 3.0f}, {3});
    const float* buffer = storage.data().data();

    loganlm::Tensor t(std::move(storage));

    EXPECT_EQ(t.storage().data().data(), buffer);
    EXPECT_EQ(t.storage().data(), (std::vector<float>{1.0f, 2.0f, 3.0f}));
}

TEST(Apply, AddsElementWiseIntoANewTensor) {
    loganlm::Tensor a(loganlm::Storage({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}));
    loganlm::Tensor b(loganlm::Storage({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, {2, 3}));
    loganlm::Add add;

    loganlm::Tensor result = loganlm::apply(add, {&a, &b});

    EXPECT_EQ(result.storage().shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(result.storage().data(),
              (std::vector<float>{11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f}));
}

TEST(Apply, LeavesInputsUnchanged) {
    loganlm::Tensor a(loganlm::Storage({1.0f, 2.0f, 3.0f}, {3}));
    loganlm::Tensor b(loganlm::Storage({10.0f, 20.0f, 30.0f}, {3}));
    loganlm::Add add;

    loganlm::Tensor result = loganlm::apply(add, {&a, &b});

    EXPECT_EQ(a.storage().data(), (std::vector<float>{1.0f, 2.0f, 3.0f}));
    EXPECT_EQ(a.storage().shape(), (std::vector<std::size_t>{3}));
    EXPECT_EQ(b.storage().data(), (std::vector<float>{10.0f, 20.0f, 30.0f}));
    EXPECT_EQ(b.storage().shape(), (std::vector<std::size_t>{3}));
}

TEST(Apply, ResultOwnsItsOwnBuffer) {
    loganlm::Tensor a(loganlm::Storage({1.0f, 2.0f, 3.0f}, {3}));
    loganlm::Tensor b(loganlm::Storage({10.0f, 20.0f, 30.0f}, {3}));
    loganlm::Add add;

    loganlm::Tensor result = loganlm::apply(add, {&a, &b});

    EXPECT_NE(result.storage().data().data(), a.storage().data().data());
    EXPECT_NE(result.storage().data().data(), b.storage().data().data());
}

TEST(Apply, ThrowsWhenTheOperationRejectsItsInputs) {
    loganlm::Tensor a(loganlm::Storage(std::vector<float>(6, 0.0f), {2, 3}));
    loganlm::Tensor b(loganlm::Storage(std::vector<float>(6, 0.0f), {3, 2}));
    loganlm::Add add;

    EXPECT_THROW((void)loganlm::apply(add, {&a, &b}), std::invalid_argument);
}

TEST(Apply, WorksThroughABaseClassReference) {
    loganlm::Tensor a(loganlm::Storage({1.0f, 2.0f}, {2}));
    loganlm::Tensor b(loganlm::Storage({3.0f, 4.0f}, {2}));
    loganlm::Add add;
    const loganlm::Operation& operation = add;

    loganlm::Tensor result = loganlm::apply(operation, {&a, &b});

    EXPECT_EQ(result.storage().data(), (std::vector<float>{4.0f, 6.0f}));
}
