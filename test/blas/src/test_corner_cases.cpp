#include <type_traits>
#include <catch2/catch.hpp>
#include <tblas.hpp>
#include "test_types.hpp"

using namespace blas;

TEMPLATE_TEST_CASE( "asum satisfies all corner cases", "[asum][BLASlv1]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    blas::size_t n = 1;
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( asum( n, x, 0 ), Error );
        CHECK_THROWS_AS( asum( n, x, -1 ), Error );
    }
    SECTION ( "n <= 0" ) {
        if( std::is_signed<blas::size_t>::value )
            CHECK( asum(-1, x, incx ) == real_t(0) );
        CHECK( asum( 0, x, incx ) == real_t(0) );
    }
}

TEMPLATE_TEST_CASE( "axpy satisfies all corner cases", "[axpy][BLASlv1]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;

    // Corner cases:
    if( std::is_signed<blas::size_t>::value ) {
    SECTION( "n = -1" ) {
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( axpy( -1, alpha, x, incx, y, incy ) );
        CHECK( (std::equal(y, y+5, ref_y)) );
        std::swap( y, ref_y );
    }}
    SECTION( "n = 0" ) {
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( axpy( 0, alpha, x, incx, y, incy ) );
        CHECK( (std::equal(y, y+5, ref_y)) );
        std::swap( y, ref_y );
    }
    SECTION( "alpha = real_t(0)" ) {
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( axpy( n, real_t(0), x, incx, y, incy ) );
        CHECK( (std::equal(y, y+5, ref_y)) );
        std::swap( y, ref_y );
    }
}

TEMPLATE_TEST_CASE( "copy satisfies all corner cases", "[copy][BLASlv1]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    blas::size_t n = 1;
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;

    // Corner cases:
    if( std::is_signed<blas::size_t>::value ) {
    SECTION( "n = -1" ) {
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( copy( -1, x, incx, y, incy ) );
        CHECK( (std::equal(y, y+5, ref_y)) );
        std::swap( y, ref_y );
    }}
    SECTION( "n = 0" ) {
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( copy( 0, x, incx, y, incy ) );
        CHECK( (std::equal(y, y+5, ref_y)) );
        std::swap( y, ref_y );
    }
}

TEMPLATE_TEST_CASE( "dot satisfies all corner cases", "[dot][BLASlv1]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    blas::size_t n = 1;
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType const y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;

    // Corner cases:
    SECTION ( "n <= 0" ) {
        if( std::is_signed<blas::size_t>::value )
            CHECK( dot(-1, x, incx, y, incy ) == real_t(0) );
        CHECK( dot( 0, x, incx, y, incy ) == real_t(0) );
    }
}

TEMPLATE_TEST_CASE( "dotu satisfies all corner cases", "[dotu][BLASlv1]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    blas::size_t n = 1;
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType const y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;

    // Corner cases:
    SECTION ( "n <= 0" ) {
        if( std::is_signed<blas::size_t>::value )
            CHECK( dotu(-1, x, incx, y, incy ) == real_t(0) );
        CHECK( dotu( 0, x, incx, y, incy ) == real_t(0) );
    }
}

TEMPLATE_TEST_CASE( "iamax satisfies all corner cases", "[iamax][BLASlv1]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    blas::size_t n = 1;
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( iamax( n, x, 0 ), Error );
        CHECK_THROWS_AS( iamax( n, x, -1 ), Error );
    }
    SECTION ( "n <= 0" ) {
        if( std::is_signed<blas::size_t>::value )
            CHECK( iamax(-1, x, incx ) == INVALID_INDEX );
        CHECK( iamax( 0, x, incx ) == INVALID_INDEX );
    }
}

TEMPLATE_TEST_CASE( "nrm2 satisfies all corner cases", "[nrm2][BLASlv1]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    blas::size_t n = 1;
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( nrm2( n, x, 0 ), Error );
        CHECK_THROWS_AS( nrm2( n, x, -1 ), Error );
    }
    SECTION ( "n <= 0" ) {
        if( std::is_signed<blas::size_t>::value )
            CHECK( nrm2(-1, x, incx ) == real_t(0) );
        CHECK( nrm2( 0, x, incx ) == real_t(0) );
    }
}

TEMPLATE_TEST_CASE( "rot satisfies all corner cases", "[rot][BLASlv1]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    blas::size_t n = 1;
    TestType x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;
    real_type<TestType> c = real_t(1);
    TestType s = real_t(1);

    // Corner cases:
    if( std::is_signed<blas::size_t>::value ) {
    SECTION( "n = -1" ) {
        TestType ref_x[5];
        std::copy( x, x+5, ref_x );
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( rot( -1, x, incx, y, incy, c, s ) );
        CHECK( (std::equal(x, x+5, ref_x) && std::equal(y, y+5, ref_y)) );
        std::swap( x, ref_x );
        std::swap( y, ref_y );
    }}
    SECTION( "n = 0" ) {
        TestType ref_x[5];
        std::copy( x, x+5, ref_x );
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( rot( 0, x, incx, y, incy, c, s ) );
        CHECK( (std::equal(x, x+5, ref_x) && std::equal(y, y+5, ref_y)) );
        std::swap( x, ref_x );
        std::swap( y, ref_y );
    }
    SECTION( "c = real_t(1); s = real_t(0)" ) {
        TestType ref_x[5];
        std::copy( x, x+5, ref_x );
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( rot( n, x, incx, y, incy, real_t(1), real_t(0) ) );
        CHECK( (std::equal(x, x+5, ref_x) && std::equal(y, y+5, ref_y)) );
        std::swap( x, ref_x );
        std::swap( y, ref_y );
    }
}

/*
TEMPLATE_TEST_CASE( "rotg satisfies all corner cases", "[rotg][BLASlv1]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    TestType a[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    TestType b[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    real_type<TestType> c[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    TestType s[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};

    // Corner cases:
} */

TEMPLATE_TEST_CASE( "rotm satisfies all corner cases", "[rotm][BLASlv1]", TEST_REAL_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    blas::size_t n = 1;
    TestType x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;
    TestType const param[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};

    // Corner cases:
    if( std::is_signed<blas::size_t>::value ) {
    SECTION( "n = -1" ) {
        TestType ref_x[5];
        std::copy( x, x+5, ref_x );
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( rotm( -1, x, incx, y, incy, param ) );
        CHECK( (std::equal(x, x+5, ref_x) && std::equal(y, y+5, ref_y)) );
        std::swap( x, ref_x );
        std::swap( y, ref_y );
    }}
    SECTION( "n = 0" ) {
        TestType ref_x[5];
        std::copy( x, x+5, ref_x );
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( rotm( 0, x, incx, y, incy, param ) );
        CHECK( (std::equal(x, x+5, ref_x) && std::equal(y, y+5, ref_y)) );
        std::swap( x, ref_x );
        std::swap( y, ref_y );
    }
}

TEMPLATE_TEST_CASE( "rotmg satisfies all corner cases", "[rotmg][BLASlv1]", TEST_REAL_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    TestType d1[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    TestType d2[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    TestType a[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    TestType b = real_t(1);
    TestType param[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};

    // Corner cases:
    SECTION ( "Throw if d1 == -1" ) {
        real_t d1Minus1 = real_t(-1);
        CHECK_THROWS_AS( rotmg( &d1Minus1, d2, a, b, param ), Error );
    }
}

TEMPLATE_TEST_CASE( "scal satisfies all corner cases", "[scal][BLASlv1]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( scal( n, alpha, x, 0 ), Error );
        CHECK_THROWS_AS( scal( n, alpha, x, -1 ), Error );
    }
    if( std::is_signed<blas::size_t>::value ) {
    SECTION( "n = -1" ) {
        TestType ref_x[5];
        std::copy( x, x+5, ref_x );
        REQUIRE_NOTHROW( scal( -1, alpha, x, incx ) );
        CHECK( (std::equal(x, x+5, ref_x)) );
        std::swap( x, ref_x );
    }}
    SECTION( "n = 0" ) {
        TestType ref_x[5];
        std::copy( x, x+5, ref_x );
        REQUIRE_NOTHROW( scal( 0, alpha, x, incx ) );
        CHECK( (std::equal(x, x+5, ref_x)) );
        std::swap( x, ref_x );
    }
}

TEMPLATE_TEST_CASE( "swap satisfies all corner cases", "[swap][BLASlv1]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    blas::size_t n = 1;
    TestType x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;

    // Corner cases:
    if( std::is_signed<blas::size_t>::value ) {
    SECTION( "n = -1" ) {
        TestType ref_x[5];
        std::copy( x, x+5, ref_x );
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( swap( -1, x, incx, y, incy ) );
        CHECK( (std::equal(x, x+5, ref_x) && std::equal(y, y+5, ref_y)) );
        std::swap( x, ref_x );
        std::swap( y, ref_y );
    }}
    SECTION( "n = 0" ) {
        TestType ref_x[5];
        std::copy( x, x+5, ref_x );
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( swap( 0, x, incx, y, incy ) );
        CHECK( (std::equal(x, x+5, ref_x) && std::equal(y, y+5, ref_y)) );
        std::swap( x, ref_x );
        std::swap( y, ref_y );
    }
}

TEMPLATE_TEST_CASE( "gemv satisfies all corner cases", "[gemv][BLASlv2]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Op trans = Op::NoTrans;
    blas::size_t m = 1;
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType beta = real_t(1);
    TestType y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( gemv( Layout(0), trans, m, n, alpha, A, lda, x, incx, beta, y, incy ), Error );
        CHECK_THROWS_AS( gemv( layout, Op(0), m, n, alpha, A, lda, x, incx, beta, y, incy ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( gemv( layout, trans, -1, n, alpha, A, lda, x, incx, beta, y, incy ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( gemv( layout, trans, m, -1, alpha, A, lda, x, incx, beta, y, incy ), Error );
        CHECK_THROWS_AS( gemv( layout, trans, m, n, alpha, A, lda, x, 0, beta, y, incy ), Error );
        CHECK_THROWS_AS( gemv( layout, trans, m, n, alpha, A, lda, x, incx, beta, y, 0 ), Error );
        CHECK_THROWS_AS( gemv( layout, trans, 2, n, alpha, A, 1, x, incx, beta, y, incy ), Error );
        CHECK_THROWS_AS( gemv( Layout::RowMajor, trans, m, 2, alpha, A, 1, x, incx, beta, y, incy ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( gemv( layout, trans, m, 0, alpha, A, lda, x, incx, beta, y, incy ) );
        CHECK( (std::equal(y, y+5, ref_y)) );
        std::swap( y, ref_y );
    }
    SECTION( "m = 0" ) {
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( gemv( layout, trans, 0, n, alpha, A, lda, x, incx, beta, y, incy ) );
        CHECK( (std::equal(y, y+5, ref_y)) );
        std::swap( y, ref_y );
    }
    SECTION( "y does not need to be set if beta = 0" ) {
        y[0] = y[1] = real_t(NAN);
        REQUIRE_NOTHROW( gemv( layout, trans, 2, 2, alpha, A, 2, x, incx, real_t(0), y, incy ) );
        CHECK( (y[0] == y[0] && y[1] == y[1]) ); // i.e., they are not NaN
        y[0] = y[1] = 1;
    }
}

TEMPLATE_TEST_CASE( "ger satisfies all corner cases", "[ger][BLASlv2]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    blas::size_t m = 1;
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType const y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;
    TestType A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( ger( Layout(0), m, n, alpha, x, incx, y, incy, A, lda ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( ger( layout, -1, n, alpha, x, incx, y, incy, A, lda ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( ger( layout, m, -1, alpha, x, incx, y, incy, A, lda ), Error );
        CHECK_THROWS_AS( ger( layout, m, n, alpha, x, 0, y, incy, A, lda ), Error );
        CHECK_THROWS_AS( ger( layout, m, n, alpha, x, incx, y, 0, A, lda ), Error );
        CHECK_THROWS_AS( ger( layout, 2, n, alpha, x, incx, y, incy, A, 1 ), Error );
        CHECK_THROWS_AS( ger( Layout::RowMajor, m, 2, alpha, x, incx, y, incy, A, 1 ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_A[5];
        std::copy( A, A+5, ref_A );
        REQUIRE_NOTHROW( ger( layout, m, 0, alpha, x, incx, y, incy, A, lda ) );
        CHECK( (std::equal(A, A+5, ref_A)) );
        std::swap( A, ref_A );
    }
    SECTION( "m = 0" ) {
        TestType ref_A[5];
        std::copy( A, A+5, ref_A );
        REQUIRE_NOTHROW( ger( layout, 0, n, alpha, x, incx, y, incy, A, lda ) );
        CHECK( (std::equal(A, A+5, ref_A)) );
        std::swap( A, ref_A );
    }
}

TEMPLATE_TEST_CASE( "geru satisfies all corner cases", "[geru][BLASlv2]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    blas::size_t m = 1;
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType const y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;
    TestType A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( geru( Layout(0), m, n, alpha, x, incx, y, incy, A, lda ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( geru( layout, -1, n, alpha, x, incx, y, incy, A, lda ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( geru( layout, m, -1, alpha, x, incx, y, incy, A, lda ), Error );
        CHECK_THROWS_AS( geru( layout, m, n, alpha, x, 0, y, incy, A, lda ), Error );
        CHECK_THROWS_AS( geru( layout, m, n, alpha, x, incx, y, 0, A, lda ), Error );
        CHECK_THROWS_AS( geru( layout, 2, n, alpha, x, incx, y, incy, A, 1 ), Error );
        CHECK_THROWS_AS( geru( Layout::RowMajor, m, 2, alpha, x, incx, y, incy, A, 1 ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_A[5];
        std::copy( A, A+5, ref_A );
        REQUIRE_NOTHROW( geru( layout, m, 0, alpha, x, incx, y, incy, A, lda ) );
        CHECK( (std::equal(A, A+5, ref_A)) );
        std::swap( A, ref_A );
    }
    SECTION( "m = 0" ) {
        TestType ref_A[5];
        std::copy( A, A+5, ref_A );
        REQUIRE_NOTHROW( geru( layout, 0, n, alpha, x, incx, y, incy, A, lda ) );
        CHECK( (std::equal(A, A+5, ref_A)) );
        std::swap( A, ref_A );
    }
}

TEMPLATE_TEST_CASE( "hemv satisfies all corner cases", "[hemv][BLASlv2]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Uplo uplo = Uplo::Upper;
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType beta = real_t(1);
    TestType y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( hemv( Layout(0), uplo, n, alpha, A, lda, x, incx, beta, y, incy ), Error );
        CHECK_THROWS_AS( hemv( layout, Uplo(0), n, alpha, A, lda, x, incx, beta, y, incy ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( hemv( layout, uplo, -1, alpha, A, lda, x, incx, beta, y, incy ), Error );
        CHECK_THROWS_AS( hemv( layout, uplo, n, alpha, A, lda, x, 0, beta, y, incy ), Error );
        CHECK_THROWS_AS( hemv( layout, uplo, n, alpha, A, lda, x, incx, beta, y, 0 ), Error );
        CHECK_THROWS_AS( hemv( layout, uplo, 2, alpha, A, 1, x, incx, beta, y, incy ), Error );
        CHECK_THROWS_AS( hemv( Layout::RowMajor, uplo, 2, alpha, A, 1, x, incx, beta, y, incy ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( hemv( layout, uplo, 0, alpha, A, lda, x, incx, beta, y, incy ) );
        CHECK( (std::equal(y, y+5, ref_y)) );
        std::swap( y, ref_y );
    }
    SECTION( "y does not need to be set if beta = 0" ) {
        y[0] = y[1] = real_t(NAN);
        REQUIRE_NOTHROW( hemv( layout, uplo, 2, alpha, A, 2, x, incx, real_t(0), y, incy ) );
        CHECK( (y[0] == y[0] && y[1] == y[1]) ); // i.e., they are not NaN
        y[0] = y[1] = 1;
    }
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of A is not referenced" ) {
        complex_t const _A[] = {{1, real_t(NAN)}, real_t(1), real_t(1), {1, real_t(NAN)}};
        REQUIRE_NOTHROW( hemv( layout, uplo, 2, alpha, _A, 2, (complex_t const *)x, incx, beta, (complex_t *)y, incy ) );
        CHECK( (y[0] == y[0] && y[1] == y[1]) ); // i.e., they are not NaN
        y[0] = y[1] = 1;
    }}
}

TEMPLATE_TEST_CASE( "her satisfies all corner cases", "[her][BLASlv2]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Uplo uplo = Uplo::Upper;
    blas::size_t n = 1;
    real_type<TestType> alpha = real_t(1);
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( her( Layout(0), uplo, n, alpha, x, incx, A, lda ), Error );
        CHECK_THROWS_AS( her( layout, Uplo(0), n, alpha, x, incx, A, lda ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( her( layout, uplo, -1, alpha, x, incx, A, lda ), Error );
        CHECK_THROWS_AS( her( layout, uplo, n, alpha, x, 0, A, lda ), Error );
        CHECK_THROWS_AS( her( layout, uplo, 2, alpha, x, incx, A, 1 ), Error );
        CHECK_THROWS_AS( her( Layout::RowMajor, uplo, 2, alpha, x, incx, A, 1 ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_A[5];
        std::copy( A, A+5, ref_A );
        REQUIRE_NOTHROW( her( layout, uplo, 0, alpha, x, incx, A, lda ) );
        CHECK( (std::equal(A, A+5, ref_A)) );
        std::swap( A, ref_A );
    }
    SECTION( "alpha = real_t(0)" ) {
        TestType ref_A[5];
        std::copy( A, A+5, ref_A );
        REQUIRE_NOTHROW( her( layout, uplo, n, real_t(0), x, incx, A, lda ) );
        CHECK( (std::equal(A, A+5, ref_A)) );
        std::swap( A, ref_A );
    }
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of A is zero" ) {
        complex_t _A[] = {{1, real_t(NAN)}, real_t(1), real_t(1), {1, real_t(NAN)}};
        REQUIRE_NOTHROW( her( layout, uplo, 2, alpha, (complex_t const *)x, incx, _A, 2 ) );
        CHECK( (_A[0] == _A[0] && _A[1] == _A[1] && _A[2] == _A[2] && _A[3] == _A[3]) ); // i.e., they are not NaN
        CHECK( (std::imag(_A[0]) == real_t(0) && std::imag(_A[3]) == real_t(0)) );
    }}
}

TEMPLATE_TEST_CASE( "her2 satisfies all corner cases", "[her2][BLASlv2]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Uplo uplo = Uplo::Upper;
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType const y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;
    TestType A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( her2( Layout(0), uplo, n, alpha, x, incx, y, incy, A, lda ), Error );
        CHECK_THROWS_AS( her2( layout, Uplo(0), n, alpha, x, incx, y, incy, A, lda ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( her2( layout, uplo, -1, alpha, x, incx, y, incy, A, lda ), Error );
        CHECK_THROWS_AS( her2( layout, uplo, n, alpha, x, 0, y, incy, A, lda ), Error );
        CHECK_THROWS_AS( her2( layout, uplo, n, alpha, x, incx, y, 0, A, lda ), Error );
        CHECK_THROWS_AS( her2( layout, uplo, 2, alpha, x, incx, y, incy, A, 1 ), Error );
        CHECK_THROWS_AS( her2( Layout::RowMajor, uplo, 2, alpha, x, incx, y, incy, A, 1 ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_A[5];
        std::copy( A, A+5, ref_A );
        REQUIRE_NOTHROW( her2( layout, uplo, 0, alpha, x, incx, y, incy, A, lda ) );
        CHECK( (std::equal(A, A+5, ref_A)) );
        std::swap( A, ref_A );
    }
    SECTION( "alpha = real_t(0)" ) {
        TestType ref_A[5];
        std::copy( A, A+5, ref_A );
        REQUIRE_NOTHROW( her2( layout, uplo, n, real_t(0), x, incx, y, incy, A, lda ) );
        CHECK( (std::equal(A, A+5, ref_A)) );
        std::swap( A, ref_A );
    }
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of A is zero" ) {
        complex_t _A[] = {{1, real_t(NAN)}, real_t(1), real_t(1), {1, real_t(NAN)}};
        REQUIRE_NOTHROW( her2( layout, uplo, 2, alpha, (complex_t const *)x, incx, y, incy, _A, 2 ) );
        CHECK( (_A[0] == _A[0] && _A[1] == _A[1] && _A[2] == _A[2] && _A[3] == _A[3]) ); // i.e., they are not NaN
        CHECK( (std::imag(_A[0]) == real_t(0) && std::imag(_A[3]) == real_t(0)) );
    }}
}

TEMPLATE_TEST_CASE( "symv satisfies all corner cases", "[symv][BLASlv2]", TEST_REAL_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Uplo uplo = Uplo::Upper;
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType beta = real_t(1);
    TestType y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( symv( Layout(0), uplo, n, alpha, A, lda, x, incx, beta, y, incy ), Error );
        CHECK_THROWS_AS( symv( layout, Uplo(0), n, alpha, A, lda, x, incx, beta, y, incy ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( symv( layout, uplo, -1, alpha, A, lda, x, incx, beta, y, incy ), Error );
        CHECK_THROWS_AS( symv( layout, uplo, n, alpha, A, lda, x, 0, beta, y, incy ), Error );
        CHECK_THROWS_AS( symv( layout, uplo, n, alpha, A, lda, x, incx, beta, y, 0 ), Error );
        CHECK_THROWS_AS( symv( layout, uplo, 2, alpha, A, 1, x, incx, beta, y, incy ), Error );
        CHECK_THROWS_AS( symv( Layout::RowMajor, uplo, 2, alpha, A, 1, x, incx, beta, y, incy ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_y[5];
        std::copy( y, y+5, ref_y );
        REQUIRE_NOTHROW( symv( layout, uplo, 0, alpha, A, lda, x, incx, beta, y, incy ) );
        CHECK( (std::equal(y, y+5, ref_y)) );
        std::swap( y, ref_y );
    }
    SECTION( "y does not need to be set if beta = 0" ) {
        y[0] = y[1] = real_t(NAN);
        REQUIRE_NOTHROW( symv( layout, uplo, 2, alpha, A, 2, x, incx, real_t(0), y, incy ) );
        CHECK( (y[0] == y[0] && y[1] == y[1]) ); // i.e., they are not NaN
        y[0] = y[1] = 1;
    }
}

TEMPLATE_TEST_CASE( "syr satisfies all corner cases", "[syr][BLASlv2]", TEST_REAL_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Uplo uplo = Uplo::Upper;
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( syr( Layout(0), uplo, n, alpha, x, incx, A, lda ), Error );
        CHECK_THROWS_AS( syr( layout, Uplo(0), n, alpha, x, incx, A, lda ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( syr( layout, uplo, -1, alpha, x, incx, A, lda ), Error );
        CHECK_THROWS_AS( syr( layout, uplo, n, alpha, x, 0, A, lda ), Error );
        CHECK_THROWS_AS( syr( layout, uplo, 2, alpha, x, incx, A, 1 ), Error );
        CHECK_THROWS_AS( syr( Layout::RowMajor, uplo, 2, alpha, x, incx, A, 1 ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_A[5];
        std::copy( A, A+5, ref_A );
        REQUIRE_NOTHROW( syr( layout, uplo, 0, alpha, x, incx, A, lda ) );
        CHECK( (std::equal(A, A+5, ref_A)) );
        std::swap( A, ref_A );
    }
    SECTION( "alpha = real_t(0)" ) {
        TestType ref_A[5];
        std::copy( A, A+5, ref_A );
        REQUIRE_NOTHROW( syr( layout, uplo, n, real_t(0), x, incx, A, lda ) );
        CHECK( (std::equal(A, A+5, ref_A)) );
        std::swap( A, ref_A );
    }
}

TEMPLATE_TEST_CASE( "syr2 satisfies all corner cases", "[syr2][BLASlv2]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Uplo uplo = Uplo::Upper;
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType const x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;
    TestType const y[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incy = 1;
    TestType A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( syr2( Layout(0), uplo, n, alpha, x, incx, y, incy, A, lda ), Error );
        CHECK_THROWS_AS( syr2( layout, Uplo(0), n, alpha, x, incx, y, incy, A, lda ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( syr2( layout, uplo, -1, alpha, x, incx, y, incy, A, lda ), Error );
        CHECK_THROWS_AS( syr2( layout, uplo, n, alpha, x, 0, y, incy, A, lda ), Error );
        CHECK_THROWS_AS( syr2( layout, uplo, n, alpha, x, incx, y, 0, A, lda ), Error );
        CHECK_THROWS_AS( syr2( layout, uplo, 2, alpha, x, incx, y, incy, A, 1 ), Error );
        CHECK_THROWS_AS( syr2( Layout::RowMajor, uplo, 2, alpha, x, incx, y, incy, A, 1 ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_A[5];
        std::copy( A, A+5, ref_A );
        REQUIRE_NOTHROW( syr2( layout, uplo, 0, alpha, x, incx, y, incy, A, lda ) );
        CHECK( (std::equal(A, A+5, ref_A)) );
        std::swap( A, ref_A );
    }
    SECTION( "alpha = real_t(0)" ) {
        TestType ref_A[5];
        std::copy( A, A+5, ref_A );
        REQUIRE_NOTHROW( syr2( layout, uplo, n, real_t(0), x, incx, y, incy, A, lda ) );
        CHECK( (std::equal(A, A+5, ref_A)) );
        std::swap( A, ref_A );
    }
}

TEMPLATE_TEST_CASE( "trmv satisfies all corner cases", "[trmv][BLASlv2]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Uplo uplo = Uplo::Upper;
    Op trans = Op::NoTrans;
    Diag diag = Diag::NonUnit;
    blas::size_t n = 1;
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    TestType x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( trmv( Layout(0), uplo, trans, diag, n, A, lda, x, incx ), Error );
        CHECK_THROWS_AS( trmv( layout, Uplo(0), trans, diag, n, A, lda, x, incx ), Error );
        CHECK_THROWS_AS( trmv( layout, uplo, Op(0), diag, n, A, lda, x, incx ), Error );
        CHECK_THROWS_AS( trmv( layout, uplo, trans, Diag(0), n, A, lda, x, incx ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( trmv( layout, uplo, trans, diag, -1, A, lda, x, incx ), Error );
        CHECK_THROWS_AS( trmv( layout, uplo, trans, diag, n, A, lda, x, 0 ), Error );
        CHECK_THROWS_AS( trmv( layout, uplo, trans, diag, 2, A, 1, x, incx ), Error );
        CHECK_THROWS_AS( trmv( Layout::RowMajor, uplo, trans, diag, 2, A, 1, x, incx ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_x[5];
        std::copy( x, x+5, ref_x );
        REQUIRE_NOTHROW( trmv( layout, uplo, trans, diag, 0, A, lda, x, incx ) );
        CHECK( (std::equal(x, x+5, ref_x)) );
        std::swap( x, ref_x );
    }
    SECTION( "Diagonal of A is not referenced if diag = 'U'" ) {
        TestType const _A[] = {real_t(NAN), real_t(1), real_t(1), real_t(NAN)};
        REQUIRE_NOTHROW( trmv( layout, uplo, trans, Diag('U'), 2, _A, 2, x, incx ) );
        CHECK( (x[0] == x[0] && x[1] == x[1]) ); // i.e., they are not NaN
        x[0] = x[1] = 1;
    }
}

TEMPLATE_TEST_CASE( "trsv satisfies all corner cases", "[trsv][BLASlv2]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Uplo uplo = Uplo::Upper;
    Op trans = Op::NoTrans;
    Diag diag = Diag::NonUnit;
    blas::size_t n = 1;
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    TestType x[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::int_t incx = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( trsv( Layout(0), uplo, trans, diag, n, A, lda, x, incx ), Error );
        CHECK_THROWS_AS( trsv( layout, Uplo(0), trans, diag, n, A, lda, x, incx ), Error );
        CHECK_THROWS_AS( trsv( layout, uplo, Op(0), diag, n, A, lda, x, incx ), Error );
        CHECK_THROWS_AS( trsv( layout, uplo, trans, Diag(0), n, A, lda, x, incx ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( trsv( layout, uplo, trans, diag, -1, A, lda, x, incx ), Error );
        CHECK_THROWS_AS( trsv( layout, uplo, trans, diag, n, A, lda, x, 0 ), Error );
        CHECK_THROWS_AS( trsv( layout, uplo, trans, diag, 2, A, 1, x, incx ), Error );
        CHECK_THROWS_AS( trsv( Layout::RowMajor, uplo, trans, diag, 2, A, 1, x, incx ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_x[5];
        std::copy( x, x+5, ref_x );
        REQUIRE_NOTHROW( trsv( layout, uplo, trans, diag, 0, A, lda, x, incx ) );
        CHECK( (std::equal(x, x+5, ref_x)) );
        std::swap( x, ref_x );
    }
    SECTION( "Diagonal of A is not referenced if diag = 'U'" ) {
        TestType const _A[] = {real_t(NAN), real_t(1), real_t(1), real_t(NAN)};
        REQUIRE_NOTHROW( trsv( layout, uplo, trans, Diag('U'), 2, _A, 2, x, incx ) );
        CHECK( (x[0] == x[0] && x[1] == x[1]) ); // i.e., they are not NaN
        x[0] = x[1] = 1;
    }
}

TEMPLATE_TEST_CASE( "gemm satisfies all corner cases", "[gemm][BLASlv3]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Op transA = Op::NoTrans;
    Op transB = Op::NoTrans;
    blas::size_t m = 1;
    blas::size_t n = 1;
    blas::size_t k = 1;
    TestType alpha = real_t(1);
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    TestType const B[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldb = 1;
    TestType beta = real_t(1);
    TestType C[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldc = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( gemm( Layout(0), transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( gemm( layout, Op(0), transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( gemm( layout, transA, Op(0), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( gemm( layout, transA, transB, -1, n, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( gemm( layout, transA, transB, m, -1, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( gemm( layout, transA, transB, m, n, -1, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( gemm( layout, transA, transB, 2, n, k, alpha, A, 1, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( gemm( layout, transA, transB, m, n, 2, alpha, A, lda, B, 1, beta, C, ldc ), Error );
        CHECK_THROWS_AS( gemm( layout, transA, transB, 2, n, k, alpha, A, lda, B, ldb, beta, C, 1 ), Error );
        CHECK_THROWS_AS( gemm( Layout::RowMajor, transA, transB, m, 2, k, alpha, A, 1, B, ldb, beta, C, ldc ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( gemm( layout, transA, transB, m, 0, k, alpha, A, lda, B, ldb, beta, C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "m = 0" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( gemm( layout, transA, transB, 0, n, k, alpha, A, lda, B, ldb, beta, C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "alpha = real_t(0); beta = real_t(1)" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( gemm( layout, transA, transB, m, n, k, real_t(0), A, lda, B, ldb, real_t(1), C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "k = 0; beta = real_t(1)" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( gemm( layout, transA, transB, m, n, 0, alpha, A, lda, B, ldb, real_t(1), C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = real_t(2)*C[0];
        REQUIRE_NOTHROW( gemm( layout, transA, transB, m, n, 0, alpha, A, lda, B, ldb, real_t(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }
}

TEMPLATE_TEST_CASE( "hemm satisfies all corner cases", "[hemm][BLASlv3]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Side side = Side::Left;
    Uplo uplo = Uplo::Upper;
    blas::size_t m = 1;
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    TestType const B[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldb = 1;
    TestType beta = real_t(1);
    TestType C[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldc = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( hemm( Layout(0), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( hemm( layout, Side(0), uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( hemm( layout, side, Uplo(0), m, n, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( hemm( layout, side, uplo, -1, n, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( hemm( layout, side, uplo, m, -1, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( hemm( layout, side, uplo, 2, n, alpha, A, 1, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( hemm( layout, side, uplo, 2, n, alpha, A, lda, B, 1, beta, C, ldc ), Error );
        CHECK_THROWS_AS( hemm( layout, side, uplo, 2, n, alpha, A, lda, B, ldb, beta, C, 1 ), Error );
        CHECK_THROWS_AS( hemm( Layout::RowMajor, side, uplo, m, 2, alpha, A, 1, B, ldb, beta, C, ldc ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( hemm( layout, side, uplo, m, 0, alpha, A, lda, B, ldb, beta, C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "m = 0" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( hemm( layout, side, uplo, 0, n, alpha, A, lda, B, ldb, beta, C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "alpha = real_t(0); beta = real_t(1)" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( hemm( layout, side, uplo, m, n, real_t(0), A, lda, B, ldb, real_t(1), C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of A is not referenced" ) {
        complex_t const _A[] = {{1, real_t(NAN)}, real_t(1), real_t(1), {1, real_t(NAN)}};
        REQUIRE_NOTHROW( hemm( layout, side, uplo, 2, 2, alpha, _A, 2, B, 2, beta, (complex_t*) C, 2 ) );
        CHECK( (C[0] == C[0] && C[1] == C[1] && C[2] == C[2] && C[3] == C[3]) ); // i.e., they are not NaN
        std::fill_n(C, 5, 1);
    }}
}

TEMPLATE_TEST_CASE( "her2k satisfies all corner cases", "[her2k][BLASlv3]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Uplo uplo = Uplo::Upper;
    Op trans = Op::NoTrans;
    blas::size_t n = 1;
    blas::size_t k = 1;
    TestType alpha = real_t(1);
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    TestType const B[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldb = 1;
    real_type<TestType> beta = real_t(1);
    TestType C[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldc = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( her2k( Layout(0), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( her2k( layout, Uplo(0), trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( her2k( layout, uplo, Op(0), n, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( her2k( layout, uplo, trans, -1, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( her2k( layout, uplo, trans, n, -1, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( her2k( layout, uplo, Op('T'), n, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( her2k( layout, uplo, trans, 2, k, alpha, A, 1, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( her2k( layout, uplo, trans, 2, k, alpha, A, lda, B, 1, beta, C, ldc ), Error );
        CHECK_THROWS_AS( her2k( layout, uplo, trans, 2, k, alpha, A, lda, B, ldb, beta, C, 1 ), Error );
        CHECK_THROWS_AS( her2k( Layout::RowMajor, uplo, trans, n, 2, alpha, A, 1, B, ldb, beta, C, ldc ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( her2k( layout, uplo, trans, 0, k, alpha, A, lda, B, ldb, beta, C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "alpha = real_t(0); beta = real_t(1)" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( her2k( layout, uplo, trans, n, k, real_t(0), A, lda, B, ldb, real_t(1), C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "k = 0; beta = real_t(1)" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( her2k( layout, uplo, trans, n, 0, alpha, A, lda, B, ldb, real_t(1), C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = real_t(2)*C[0];
        REQUIRE_NOTHROW( her2k( layout, uplo, trans, n, 0, alpha, A, lda, B, ldb, real_t(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of C is zero" ) {
        complex_t _C[] = {{1, real_t(NAN)}, real_t(1), real_t(1), {1, real_t(NAN)}};
        REQUIRE_NOTHROW( her2k( layout, uplo, trans, 2, 2, alpha, A, 2, B, 2, beta, _C, 2 ) );
        CHECK( (_C[0] == _C[0] && _C[1] == _C[1] && _C[2] == _C[2] && _C[3] == _C[3]) ); // i.e., they are not NaN
    }}
}

TEMPLATE_TEST_CASE( "herk satisfies all corner cases", "[herk][BLASlv3]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Uplo uplo = Uplo::Upper;
    Op trans = Op::NoTrans;
    blas::size_t n = 1;
    blas::size_t k = 1;
    real_type<TestType> alpha = real_t(1);
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    real_type<TestType> beta = real_t(1);
    TestType C[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldc = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( herk( Layout(0), uplo, trans, n, k, alpha, A, lda, beta, C, ldc ), Error );
        CHECK_THROWS_AS( herk( layout, Uplo(0), trans, n, k, alpha, A, lda, beta, C, ldc ), Error );
        CHECK_THROWS_AS( herk( layout, uplo, Op(0), n, k, alpha, A, lda, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( herk( layout, uplo, trans, -1, k, alpha, A, lda, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( herk( layout, uplo, trans, n, -1, alpha, A, lda, beta, C, ldc ), Error );
        CHECK_THROWS_AS( herk( layout, uplo, Op('T'), n, k, alpha, A, lda, beta, C, ldc ), Error );
        CHECK_THROWS_AS( herk( layout, uplo, trans, 2, k, alpha, A, 1, beta, C, ldc ), Error );
        CHECK_THROWS_AS( herk( layout, uplo, trans, 2, k, alpha, A, lda, beta, C, 1 ), Error );
        CHECK_THROWS_AS( herk( Layout::RowMajor, uplo, trans, n, 2, alpha, A, 1, beta, C, ldc ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( herk( layout, uplo, trans, 0, k, alpha, A, lda, beta, C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "alpha = real_t(0); beta = real_t(1)" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( herk( layout, uplo, trans, n, k, real_t(0), A, lda, real_t(1), C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "k = 0; beta = real_t(1)" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( herk( layout, uplo, trans, n, 0, alpha, A, lda, real_t(1), C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = real_t(2)*C[0];
        REQUIRE_NOTHROW( herk( layout, uplo, trans, n, 0, alpha, A, lda, real_t(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of C is zero" ) {
        complex_t _C[] = {{1, real_t(NAN)}, real_t(1), real_t(1), {1, real_t(NAN)}};
        REQUIRE_NOTHROW( herk( layout, uplo, trans, 2, 2, alpha, A, 2, beta, _C, 2 ) );
        CHECK( (_C[0] == _C[0] && _C[1] == _C[1] && _C[2] == _C[2] && _C[3] == _C[3]) ); // i.e., they are not NaN
    }}
}

TEMPLATE_TEST_CASE( "symm satisfies all corner cases", "[symm][BLASlv3]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Side side = Side::Left;
    Uplo uplo = Uplo::Upper;
    blas::size_t m = 1;
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    TestType const B[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldb = 1;
    TestType beta = real_t(1);
    TestType C[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldc = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( symm( Layout(0), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( symm( layout, Side(0), uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( symm( layout, side, Uplo(0), m, n, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( symm( layout, side, uplo, -1, n, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( symm( layout, side, uplo, m, -1, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( symm( layout, side, uplo, 2, n, alpha, A, 1, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( symm( layout, side, uplo, 2, n, alpha, A, lda, B, 1, beta, C, ldc ), Error );
        CHECK_THROWS_AS( symm( layout, side, uplo, 2, n, alpha, A, lda, B, ldb, beta, C, 1 ), Error );
        CHECK_THROWS_AS( symm( Layout::RowMajor, side, uplo, m, 2, alpha, A, 1, B, ldb, beta, C, ldc ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( symm( layout, side, uplo, m, 0, alpha, A, lda, B, ldb, beta, C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "m = 0" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( symm( layout, side, uplo, 0, n, alpha, A, lda, B, ldb, beta, C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "alpha = real_t(0); beta = real_t(1)" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( symm( layout, side, uplo, m, n, real_t(0), A, lda, B, ldb, real_t(1), C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
}

TEMPLATE_TEST_CASE( "syr2k satisfies all corner cases", "[syr2k][BLASlv3]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Uplo uplo = Uplo::Upper;
    Op trans = Op::NoTrans;
    blas::size_t n = 1;
    blas::size_t k = 1;
    TestType alpha = real_t(1);
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    TestType const B[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldb = 1;
    TestType beta = real_t(1);
    TestType C[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldc = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( syr2k( Layout(0), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( syr2k( layout, Uplo(0), trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( syr2k( layout, uplo, Op(0), n, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( syr2k( layout, uplo, trans, -1, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( syr2k( layout, uplo, trans, n, -1, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( syr2k( layout, uplo, Op('C'), n, k, alpha, A, lda, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( syr2k( layout, uplo, trans, 2, k, alpha, A, 1, B, ldb, beta, C, ldc ), Error );
        CHECK_THROWS_AS( syr2k( layout, uplo, trans, 2, k, alpha, A, lda, B, 1, beta, C, ldc ), Error );
        CHECK_THROWS_AS( syr2k( layout, uplo, trans, 2, k, alpha, A, lda, B, ldb, beta, C, 1 ), Error );
        CHECK_THROWS_AS( syr2k( Layout::RowMajor, uplo, trans, n, 2, alpha, A, 1, B, ldb, beta, C, ldc ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( syr2k( layout, uplo, trans, 0, k, alpha, A, lda, B, ldb, beta, C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "alpha = real_t(0); beta = real_t(1)" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( syr2k( layout, uplo, trans, n, k, real_t(0), A, lda, B, ldb, real_t(1), C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "k = 0; beta = real_t(1)" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( syr2k( layout, uplo, trans, n, 0, alpha, A, lda, B, ldb, real_t(1), C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = real_t(2)*C[0];
        REQUIRE_NOTHROW( syr2k( layout, uplo, trans, n, 0, alpha, A, lda, B, ldb, real_t(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }
}

TEMPLATE_TEST_CASE( "syrk satisfies all corner cases", "[syrk][BLASlv3]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Uplo uplo = Uplo::Upper;
    Op trans = Op::NoTrans;
    blas::size_t n = 1;
    blas::size_t k = 1;
    TestType alpha = real_t(1);
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    TestType beta = real_t(1);
    TestType C[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldc = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( syrk( Layout(0), uplo, trans, n, k, alpha, A, lda, beta, C, ldc ), Error );
        CHECK_THROWS_AS( syrk( layout, Uplo(0), trans, n, k, alpha, A, lda, beta, C, ldc ), Error );
        CHECK_THROWS_AS( syrk( layout, uplo, Op(0), n, k, alpha, A, lda, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( syrk( layout, uplo, trans, -1, k, alpha, A, lda, beta, C, ldc ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( syrk( layout, uplo, trans, n, -1, alpha, A, lda, beta, C, ldc ), Error );
        CHECK_THROWS_AS( syrk( layout, uplo, Op('C'), n, k, alpha, A, lda, beta, C, ldc ), Error );
        CHECK_THROWS_AS( syrk( layout, uplo, trans, 2, k, alpha, A, 1, beta, C, ldc ), Error );
        CHECK_THROWS_AS( syrk( layout, uplo, trans, 2, k, alpha, A, lda, beta, C, 1 ), Error );
        CHECK_THROWS_AS( syrk( Layout::RowMajor, uplo, trans, n, 2, alpha, A, 1, beta, C, ldc ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( syrk( layout, uplo, trans, 0, k, alpha, A, lda, beta, C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "alpha = real_t(0); beta = real_t(1)" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( syrk( layout, uplo, trans, n, k, real_t(0), A, lda, real_t(1), C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "k = 0; beta = real_t(1)" ) {
        TestType ref_C[5];
        std::copy( C, C+5, ref_C );
        REQUIRE_NOTHROW( syrk( layout, uplo, trans, n, 0, alpha, A, lda, real_t(1), C, ldc ) );
        CHECK( (std::equal(C, C+5, ref_C)) );
        std::swap( C, ref_C );
    }
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = real_t(2)*C[0];
        REQUIRE_NOTHROW( syrk( layout, uplo, trans, n, 0, alpha, A, lda, real_t(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }
}

TEMPLATE_TEST_CASE( "trmm satisfies all corner cases", "[trmm][BLASlv3]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Side side = Side::Left;
    Uplo uplo = Uplo::Upper;
    Op trans = Op::NoTrans;
    Diag diag = Diag::NonUnit;
    blas::size_t m = 1;
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    TestType B[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldb = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( trmm( Layout(0), side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb ), Error );
        CHECK_THROWS_AS( trmm( layout, Side(0), uplo, trans, diag, m, n, alpha, A, lda, B, ldb ), Error );
        CHECK_THROWS_AS( trmm( layout, side, Uplo(0), trans, diag, m, n, alpha, A, lda, B, ldb ), Error );
        CHECK_THROWS_AS( trmm( layout, side, uplo, Op(0), diag, m, n, alpha, A, lda, B, ldb ), Error );
        CHECK_THROWS_AS( trmm( layout, side, uplo, trans, Diag(0), m, n, alpha, A, lda, B, ldb ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( trmm( layout, side, uplo, trans, diag, -1, n, alpha, A, lda, B, ldb ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( trmm( layout, side, uplo, trans, diag, m, -1, alpha, A, lda, B, ldb ), Error );
        CHECK_THROWS_AS( trmm( layout, side, uplo, trans, diag, 2, n, alpha, A, 1, B, ldb ), Error );
        CHECK_THROWS_AS( trmm( layout, side, uplo, trans, diag, 2, n, alpha, A, lda, B, 1 ), Error );
        CHECK_THROWS_AS( trmm( Layout::RowMajor, side, uplo, trans, diag, m, 2, alpha, A, 1, B, ldb ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_B[5];
        std::copy( B, B+5, ref_B );
        REQUIRE_NOTHROW( trmm( layout, side, uplo, trans, diag, m, 0, alpha, A, lda, B, ldb ) );
        CHECK( (std::equal(B, B+5, ref_B)) );
        std::swap( B, ref_B );
    }
    SECTION( "m = 0" ) {
        TestType ref_B[5];
        std::copy( B, B+5, ref_B );
        REQUIRE_NOTHROW( trmm( layout, side, uplo, trans, diag, 0, n, alpha, A, lda, B, ldb ) );
        CHECK( (std::equal(B, B+5, ref_B)) );
        std::swap( B, ref_B );
    }
}

TEMPLATE_TEST_CASE( "trsm satisfies all corner cases", "[trsm][BLASlv3]", TEST_TYPES ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:
    Layout layout = Layout::ColMajor;
    Side side = Side::Left;
    Uplo uplo = Uplo::Upper;
    Op trans = Op::NoTrans;
    Diag diag = Diag::NonUnit;
    blas::size_t m = 1;
    blas::size_t n = 1;
    TestType alpha = real_t(1);
    TestType const A[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t lda = 1;
    TestType B[] = {real_t(1), real_t(1), real_t(1), real_t(1), real_t(1)};
    blas::size_t ldb = 1;

    // Corner cases:
    SECTION( "Throw Error Tests" ) {
        CHECK_THROWS_AS( trsm( Layout(0), side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb ), Error );
        CHECK_THROWS_AS( trsm( layout, Side(0), uplo, trans, diag, m, n, alpha, A, lda, B, ldb ), Error );
        CHECK_THROWS_AS( trsm( layout, side, Uplo(0), trans, diag, m, n, alpha, A, lda, B, ldb ), Error );
        CHECK_THROWS_AS( trsm( layout, side, uplo, Op(0), diag, m, n, alpha, A, lda, B, ldb ), Error );
        CHECK_THROWS_AS( trsm( layout, side, uplo, trans, Diag(0), m, n, alpha, A, lda, B, ldb ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( trsm( layout, side, uplo, trans, diag, -1, n, alpha, A, lda, B, ldb ), Error );
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( trsm( layout, side, uplo, trans, diag, m, -1, alpha, A, lda, B, ldb ), Error );
        CHECK_THROWS_AS( trsm( layout, side, uplo, trans, diag, 2, n, alpha, A, 1, B, ldb ), Error );
        CHECK_THROWS_AS( trsm( layout, side, uplo, trans, diag, 2, n, alpha, A, lda, B, 1 ), Error );
        CHECK_THROWS_AS( trsm( Layout::RowMajor, side, uplo, trans, diag, m, 2, alpha, A, 1, B, ldb ), Error );
    }
    SECTION( "n = 0" ) {
        TestType ref_B[5];
        std::copy( B, B+5, ref_B );
        REQUIRE_NOTHROW( trsm( layout, side, uplo, trans, diag, m, 0, alpha, A, lda, B, ldb ) );
        CHECK( (std::equal(B, B+5, ref_B)) );
        std::swap( B, ref_B );
    }
    SECTION( "m = 0" ) {
        TestType ref_B[5];
        std::copy( B, B+5, ref_B );
        REQUIRE_NOTHROW( trsm( layout, side, uplo, trans, diag, 0, n, alpha, A, lda, B, ldb ) );
        CHECK( (std::equal(B, B+5, ref_B)) );
        std::swap( B, ref_B );
    }
}
