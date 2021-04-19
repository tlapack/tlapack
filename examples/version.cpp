#include <iostream>
#include "config/version.h"

int main(int argc, char const *argv[])
{
    std::cout << "T-LAPACK version "
    << TLAPACK_VERSION_MAJOR << "."
    << TLAPACK_VERSION_MINOR << "."
    << TLAPACK_VERSION_PATCH
    << std::endl;
    return 0;
}