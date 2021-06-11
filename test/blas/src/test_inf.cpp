// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch.hpp>
#include <tblas.hpp>
#include "test_types.hpp"
#include <limits>

using namespace blas;

TEMPLATE_TEST_CASE( "iamax returns the first inf when there is no NaN",
                    "[iamax][BLASlv1][Inf]", TEST_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants:
    const real_t huge = std::numeric_limits<real_t>::max();
    const real_t inf = ( std::numeric_limits<real_t>::has_infinity )
      ? std::numeric_limits<real_t>::infinity()
      : real_t(1.0)/real_t(0.0);
    
    // Tests:
    { TestType const x[] = {-inf, inf, inf};
      CHECK( iamax( 3, x, 1 ) == 0 ); }
    { TestType const x[] = {huge, inf, -inf};
      CHECK( iamax( 3, x, 1 ) == 1 ); }
    { TestType const x[] = {huge, huge, inf};
      CHECK( iamax( 3, x, 1 ) == 2 ); }
}

TEMPLATE_TEST_CASE( "complex iamax returns the first inf when there is no NaN",
                    "[iamax][BLASlv1][Inf]", TEST_CPLX_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants:
    const TestType huge = complex_type<TestType>(
        std::numeric_limits<real_t>::max(),
        std::numeric_limits<real_t>::max()
    );
    const real_t rinf = ( std::numeric_limits<real_t>::has_infinity )
      ? std::numeric_limits<real_t>::infinity()
      : real_t(1.0)/real_t(0.0);
    const TestType cinf( rinf, rinf );
    
    // Tests:
    { TestType const x[] = {-rinf, cinf, rinf};
      CHECK( iamax( 3, x, 1 ) == 0 ); }
    { TestType const x[] = {-cinf, rinf, cinf};
      CHECK( iamax( 3, x, 1 ) == 0 ); }
    { TestType const x[] = {huge, rinf, -cinf};
      CHECK( iamax( 3, x, 1 ) == 1 ); }
    { TestType const x[] = {huge, huge, cinf};
      CHECK( iamax( 3, x, 1 ) == 2 ); }
}