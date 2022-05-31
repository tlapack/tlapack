#ifndef TRTRI_RECURSIVE
#define TRTRI_RECUSIVE

#include "trtri_v0.h"

template <typename matrix_t>
int trtri_recursive(matrix_t & C, const tlapack::Uplo & uplo){

    using T = tlapack::type_t<matrix_t>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    using real_t = tlapack::real_type<T>;


    const idx_t n = tlapack::nrows(C);

    idx_t n0 = n / 2;

    if ( n == 1 ){

        C(0,0) = real_t(1) / C(0,0);

    } else {

        if (uplo == tlapack::Uplo::Lower){

            auto C00 = tlapack::slice( C, range(0,n0), range(0,n0) );
            auto C10 = tlapack::slice( C, range(n0,n), range(0,n0) );
            auto C11 = tlapack::slice( C, range(n0,n), range(n0,n) );

            tlapack::trsm(tlapack::Side::Right, tlapack::Uplo::Lower, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, T(-1), C00, C10);
            tlapack::trsm(tlapack::Side::Left, tlapack::Uplo::Lower, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, T(+1), C11, C10);
            trtri_recursive( C00, tlapack::Uplo::Lower);
            trtri_recursive( C11, tlapack::Uplo::Lower);

            // trtri_recursive( C00, tlapack::Uplo::Lower);
            // trtri_recursive( C11, tlapack::Uplo::Lower);        
            // tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Lower, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, T(-1), C00, C10);
            // tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Lower, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, T(+1), C11, C10);

        } else {

            auto C00 = tlapack::slice( C, range(0,n0), range(0,n0) );
            auto C01 = tlapack::slice( C, range(0,n0), range(n0,n) );
            auto C11 = tlapack::slice( C, range(n0,n), range(n0,n) );

            tlapack::trsm(tlapack::Side::Left, tlapack::Uplo::Upper, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, T(-1), C00, C01);
            tlapack::trsm(tlapack::Side::Right, tlapack::Uplo::Upper, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, T(+1), C11, C01);
            trtri_recursive( C00, tlapack::Uplo::Upper);
            trtri_recursive( C11, tlapack::Uplo::Upper);

            // trtri_recursive( C00, tlapack::Uplo::Upper);
            // trtri_recursive( C11, tlapack::Uplo::Upper);        
            // tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Upper, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, T(-1), C00, C01);
            // tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, T(+1), C11, C01);

        }
    }
    
    return 0;
}

    #endif