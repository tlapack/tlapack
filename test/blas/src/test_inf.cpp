#include <catch2/catch.hpp>
#include <blas.hpp>
#include "test_types.hpp"
#include <limits>

using namespace blas;

TEMPLATE_TEST_CASE( "iamax returns the first inf when there is no NaN",
                    "[iamax][BLASlv1][Inf]", TEST_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants:
    const real_t huge = std::numeric_limits<real_t>::max();
    const real_t inf  = 2*huge;
    
    // Tests:
    { TestType const x[] = {-inf, inf, inf};
      CHECK( iamax( 3, x, 1 ) == 0 ); }
    { TestType const x[] = {huge, inf, -inf};
      CHECK( iamax( 3, x, 1 ) == 1 ); }
    { TestType const x[] = {huge, huge, inf};
      CHECK( iamax( 3, x, 1 ) == 2 ); }
}

TEMPLATE_TEST_CASE( "iamax<complex> returns the first inf when there is no NaN",
                    "[iamax][BLASlv1][Inf]", TEST_CPLX_TYPES ) {
    using real_t = real_type<TestType>;

    // Constants:
    const TestType huge = complex_type<TestType>(
        std::numeric_limits<real_t>::max(),
        std::numeric_limits<real_t>::max()
    );
    const TestType rinf( 2*std::numeric_limits<real_t>::max(), 0 );
    const TestType cinf( 0, 2*std::numeric_limits<real_t>::max() );
    
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