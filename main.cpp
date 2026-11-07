#include <iostream>

#include "loganlm/version.h"

int main() {
    std::cout << "LoganLM " << loganlm::version() << std::endl;
    return 0;
}
