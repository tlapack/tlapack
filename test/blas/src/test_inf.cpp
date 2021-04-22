#include <catch2/catch.hpp>
#include <blas.hpp>
#include "test_types.hpp"
#include <limits>

using namespace blas;

TEMPLATE_TEST_CASE( "iamax returns the first inf when there is no NaN",
                    "[iamax][BLASlv1]", TEST_TYPES ) {
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