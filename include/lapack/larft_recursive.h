// header guard
#ifndef LARFT_RECURSIVE
#define LARFT_RECURSIVE

#include "base/utils.hpp"
#include "base/types.hpp"
#include "tblas.hpp"

namespace tlapack{

/*
* @param[in] A
* @param[in] tau
* @param [in, out] T
*/


template <typename matrix_t, typename vector_t> // we need two argus because one for R&C and the other for the values of the entries
int larft_recursive(matrix_t &A, vector_t tau, matrix_t &TTT )
{
    // declaring type alias for indexes
    using idx_t = tlapack::size_type<matrix_t>;
    //using tlapack::conj;
    using range = std::pair<idx_t, idx_t>;
    using T = tlapack::type_t<matrix_t>;

    using real_t = tlapack::real_type<T>;
    

    
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if( n == 1){
        TTT(0,0) = tau[0];
    } else {
// assigning the blocks 
    idx_t n0 = n / 2;

    auto Av0 = tlapack::slice( A, range(0,m), range(0,n0) );
    auto Av1 = tlapack::slice( A, range(n0,m), range(n0,n) );
    auto T00 = tlapack::slice( TTT, range(0,n0), range(0,n0) );
    auto T11 = tlapack::slice( TTT, range(n0,n), range(n0,n) );
    auto T01 = tlapack::slice( TTT, range(0,n0), range(n0,n) );


    // slicing tau (one dimention)
    auto tau0 = tlapack::slice( tau, range(0,n0) );
    auto tau1 = tlapack::slice( tau, range(n0,n) );

    // calling the larft fun to compute T
    // tlapack::larft(tlapack::Direction::Forward, tlapack::columnwise_storage, Av0, tau0, T00);
    // tlapack::larft(tlapack::Direction::Forward, tlapack::columnwise_storage, Av1, tau1, T11);

    larft_recursive( Av0, tau0, T00);
    larft_recursive( Av1, tau1, T11);

    // Step 1
    auto A10 = tlapack::slice(A, range(n0, n), range(0, n0));
    for (idx_t i = 0; i < n0; ++i) {
        for (idx_t j = 0; j < n-n0; ++j)
            T01(i,j) = conj(A10(j,i));
    }

    // step 2
    auto A11 = tlapack::slice(A, range(n0, n), range(n0, n));
    tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Lower, tlapack::Op::NoTrans, tlapack::Diag::Unit, real_t(1), A11, T01 );

    //step 3
    auto A20 = tlapack::slice(A, range(n, m), range(0, n0));
    auto A21 = tlapack::slice(A, range(n, m), range(n0, n));
    tlapack::gemm( tlapack:: Op::ConjTrans, tlapack:: Op::NoTrans, T(1), A20, A21, T(1), T01 );

    //step 4
    tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, real_t(-1), T11, T01 );

    //step 5
    tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Upper, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, real_t(1), T00, T01 );
    }
    return 0;


}

}

#endif // LARFT_RECURSIVE



