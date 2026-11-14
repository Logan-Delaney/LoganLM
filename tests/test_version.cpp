#include <gtest/gtest.h>

#include "loganlm/version.h"

TEST(Version, IsSet) {
    EXPECT_STREQ(loganlm::version(), "0.1.0");
}