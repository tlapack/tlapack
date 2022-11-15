#ifndef TLAPACK_LARFT_RECURSIVE_HH
#define TLAPACK_LARFT_RECURSIVE_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/trmm.hpp"

namespace tlapack{

/** Forms the triangular factor T of a block reflector H of order n,
 * which is defined as a product of k elementary reflectors.
 *
 * If direction = Direction::Forward,  H = H_1 H_2 ... H_k and T is upper triangular.
 * If direction = Direction::Backward, H = H_k ... H_2 H_1 and T is lower triangular.
 *
 * If storeMode = StoreV::Columnwise, the vector which defines the elementary
 * reflector H(i) is stored in the i-th column of the array V, and
 *
 *               H  =  I - V * T * V'
 *
 * The shape of the matrix V and the storage of the vectors which define
 * the H(i) is best illustrated by the following example with n = 5 and
 * k = 3. The elements equal to 1 are not stored. The rest of the
 * array is not used.
 *
 *     direction = Forward and          direction = Forward and
 *     storeMode = Columnwise:             storeMode = Rowwise:
 *
 *     V = (  1       )                 V = (  1 v1 v1 v1 v1 )
 *         ( v1  1    )                     (     1 v2 v2 v2 )
 *         ( v1 v2  1 )                     (        1 v3 v3 )
 *         ( v1 v2 v3 )
 *         ( v1 v2 v3 )
 *
 *     direction = Backward and         direction = Backward and
 *     storeMode = Columnwise:             storeMode = Rowwise:
 *
 *     V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
 *         ( v1 v2 v3 )                     ( v2 v2 v2  1    )
 *         (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
 *         (     1 v3 )
 *         (        1 )
 * 
 * @tparam direction_t Either Direction or any class that implements `operator Direction()`.
 * @tparam storage_t Either StoreV or any class that implements `operator StoreV()`.
 * 
 * @param[in] direction
 *     Indicates how H is formed from a product of elementary reflectors.
 *     - Direction::Forward:  $H = H(1) H(2) ... H(k)$.
 *     - Direction::Backward: $H = H(k) ... H(2) H(1)$.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 * 
 * @param[in] V
 *     - If storeMode = StoreV::Columnwise: n-by-k matrix V.
 *     - If storeMode = StoreV::Rowwise:    k-by-n matrix V.
 *
 * @param[in] tau Vector of length k containing the scalar factors
 *      of the elementary reflectors H.
 * 
 * @param[out] T Matrix of size k-by-k containing the triangular factors
 *      of the block reflector.
 *     - Direction::Forward:  T is upper triangular.
 *     - Direction::Backward: T is lower triangular.
 * 
 * @ingroup auxiliary
 */
template <class direction_t, class storage_t,typename matrix_t, typename vector_t> // we need two argus because one for R&C and the other for the values of the entries
int larft_recursive(direction_t direction, storage_t storeMode,matrix_t &A, vector_t tau, matrix_t &TTT )
{
    // declaring type alias for indexes
    using idx_t = tlapack::size_type<matrix_t>;
    //using tlapack::conj;
    using range = std::pair<idx_t, idx_t>;
    using T = tlapack::type_t<matrix_t>;

    using real_t = tlapack::real_type<T>;

    // check arguments
    tlapack_check_false(    direction != Direction::Backward &&
                        direction != Direction::Forward );
    tlapack_check_false(    storeMode != StoreV::Columnwise &&
                        storeMode != StoreV::Rowwise );
    

    /// TODO:
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    const idx_t k = std::min(m,n);
    
    if( k == 1){
            TTT(0,0) = tau[0];
            return 0;
    }
    
    idx_t n0 = k / 2;
    
    
    // case1
    if(direction == Direction::Forward && storeMode == StoreV::Columnwise){
        //split T into four parts
        
        auto T00 = slice( TTT, range(0,n0), range(0,n0) );
        auto T11 = slice( TTT, range(n0,k), range(n0,k) );

        // slicing tau
        auto tau0 = slice( tau, range(0,n0) );
        auto tau1 = slice( tau, range(n0,k) );
        // in this case T01 will be nonzero 
        auto T01 = slice( TTT, range(0,n0), range(n0,k) );
        

        auto Av0 = slice( A, range(0,m), range(0,n0) );
        auto Av1 = slice( A, range(n0,m), range(n0,n) );

        // calling the larft fun to compute T
        // tlapack::larft(tlapack::Direction::Forward, tlapack::columnwise_storage, Av0, tau0, T00);
        // tlapack::larft(tlapack::Direction::Forward, tlapack::columnwise_storage, Av1, tau1, T11);

        larft_recursive( direction ,storeMode, Av0, tau0, T00);
        larft_recursive( direction ,storeMode,Av1, tau1, T11);

        // Step 1 ---- (V10)^H -> T01
        auto A10 = slice(A, range(n0, k), range(0, n0));
        // auto T01 = slice( TTT, range(0,n0), range(n0,k) );
        for (idx_t i = 0; i < n0; ++i) {
            for (idx_t j = 0; j < k-n0; ++j)
                T01(i,j) = conj(A10(j,i));

        }

        // step 2--- T01 * V11 -> T01
        auto A11 = slice(A, range(n0, k), range(n0, k));
        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Lower, tlapack::Op::NoTrans, tlapack::Diag::Unit, real_t(1), A11, T01 );

        //step 3 --- [T01 + ((V20)^H * V21)] -. T01
        if(m > n){
            auto A20 = slice(A, range(n, m), range(0, n0));
            auto A21 = slice(A, range(n, m), range(n0, n));
            tlapack::gemm( tlapack:: Op::ConjTrans, tlapack:: Op::NoTrans, T(1), A20, A21, T(1), T01 );
        }

        //step 4 --- (- T01) * T11 -> T01
        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, real_t(-1), T11, T01 );

        //step 5 --T00 * T01  -> T01
        tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Upper, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, real_t(1), T00, T01 );
             
    }
    
    //backward column wise case
    if(direction == Direction::Backward && storeMode == StoreV::Columnwise){

        auto Av0 = slice( A, range(0,m-k+n0), range(0,n0) );
        auto Av1 = slice( A, range(0,m), range(n0,k) );


        auto T00 = slice( TTT, range(0,n0), range(0,n0) );
        auto T11 = slice( TTT, range(n0,k), range(n0,k) );

        // slicing tau
        auto tau0 = slice( tau, range(0,n0) );
        auto tau1 = slice( tau, range(n0,k) );
        
        // in this case T10 will be nonzero, (k-n0)-by-n0
        auto T10 = slice( TTT, range(n0,k), range(0,n0) );

        

        // two recursive calls on left half and right half
        // larft_recursive( direction ,storeMode, Av0, tau0, T00);
        // larft_recursive( direction ,storeMode,Av1, tau1, T11);
        larft( direction ,storeMode, Av0, tau0, T00);
        larft( direction ,storeMode,Av1, tau1, T11);


        // Step 1 ---- (V01)^H -> T01
        auto A11 = slice(A, range(m-k, m-k+n0), range(n0, k));
        // auto A11 = slice(A, range(m-k, m-n0), range(n0, k));
        //auto T10 = slice( TTT, range(n0,k), range(0,n0) );
        for (idx_t i = 0; i < k-n0; ++i) {
            for (idx_t j = 0; j < n0; ++j)
                T10(i,j) = conj(A11(j,i));
        }

        // step 2--- T10 * V10 -> T10
        auto A10 = slice(A, range(m-k, m-k+n0), range(0, n0));
        // auto A10 = slice(A, range(m-k, m-n0), range(0, n0));
        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper, tlapack::Op::NoTrans, tlapack::Diag::Unit, real_t(1), A10, T10 );

        if(m > n){
            auto A00 = slice(A, range(0, m-k), range(0, n0));
            auto A01 = slice(A, range(0, m-k), range(n0, k));
            
            // auto A20 = slice(A, range(n, m), range(0, n0));
            // auto A21 = slice(A, range(n, m), range(n0, n));
            tlapack::gemm( tlapack:: Op::ConjTrans, tlapack:: Op::NoTrans, T(1), A01, A00, T(1), T10 );
        }

        //step 4 ---  T11 * (- T10) -> T10
        tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Lower, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, real_t(-1), T11, T10 );

        //step 4 ---   (T10) * T00 -> T10
        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Lower, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, real_t(1), T00, T10 );
        
    }
    
 /*   direction = Forward and
 *     storeMode = Columnwise:             storeMode = Rowwise:
 *
 *     V = (  1       )                 V = (  1 v1 v1 v1 v1 )
 *         ( v1  1    )                     (     1 v2 v2 v2 )
 *         ( v1 v2  1 )                     (        1 v3 v3 )
    
 */ 
    if(direction == Direction::Forward && storeMode == StoreV::Rowwise){
        
        //split T into four parts
        auto T00 = slice( TTT, range(0,n0), range(0,n0) );
        auto T11 = slice( TTT, range(n0,k), range(n0,k) );

        // slicing tau
        auto tau0 = slice( tau, range(0,n0) );
        auto tau1 = slice( tau, range(n0,k) );
        // in this case T01 will be nonzero 
        
        auto T01 = slice( TTT, range(0,n0), range(n0,k) );
        

        // auto Av0 = slice( A, range(0,m), range(0,n0) );
        // auto Av1 = slice( A, range(n0,m), range(n0,n) );

        auto Av0 = slice( A, range(0,n0), range(0,n) );
        auto Av1 = slice( A, range(n0,m), range(n0,n) );

        // calling the larft fun to compute T
        // tlapack::larft(tlapack::Direction::Forward, tlapack::columnwise_storage, Av0, tau0, T00);
        // tlapack::larft(tlapack::Direction::Forward, tlapack::columnwise_storage, Av1, tau1, T11);

        larft_recursive( direction ,storeMode, Av0, tau0, T00);
        larft_recursive( direction ,storeMode,Av1, tau1, T11);

        // Step 1 ---- (V10)^H -> T01
        auto A01 = slice(A, range(0, n0) , range(n0, k));
        // auto T01 = slice( TTT, range(0,n0), range(n0,k) );
        for (idx_t i = 0; i < n0; ++i) {
            for (idx_t j = 0; j < k-n0; ++j)
                T01(i,j) = A01(i,j);

        }

        // step 2--- T01 * V11 -> T01
        auto A11 = slice(A, range(n0, k), range(n0, k));
        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper, tlapack::Op::ConjTrans, tlapack::Diag::Unit, real_t(1), A11, T01 );

        //step 3 --- [T01 + ((V20)^H * V21)] -. T01
        if(m < n){
            // auto A20 = slice(A, range(n, m), range(0, n0));
            // auto A21 = slice(A, range(n, m), range(n0, n));
            
            auto A02 = slice(A, range(0, n0),range(m, n));
            auto A12 = slice(A, range(n0, m),range(m, n));
            tlapack::gemm(tlapack:: Op::NoTrans, tlapack:: Op::ConjTrans, T(1), A02, A12, T(1), T01 );
        }

        //step 4 --- (- T01) * T11 -> T01
        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, real_t(-1), T11, T01 );

        //step 5 --T00 * T01  -> T01
        tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Upper, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, real_t(1), T00, T01 );
             
    }
    
    //backward column wise case
//     *     direction = Backward and         direction = Backward and
//  *     storeMode = Columnwise:             storeMode = Rowwise:
//  *
//  *     V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
//  *         ( v1 v2 v3 )                     ( v2 v2 v2  1    )
//  *         (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
//  *         (     1 v3 )
//  *         (        1 )
    if(direction == Direction::Backward && storeMode == StoreV::Rowwise){
        
        auto Av0 = slice( A, range(0,n0) , range(0,n-k+n0));
        auto Av1 = slice( A,range(n0,k) ,range(0,n));


        auto T00 = slice( TTT, range(0,n0), range(0,n0) );
        auto T11 = slice( TTT, range(n0,k), range(n0,k) );

        // slicing tau
        auto tau0 = slice( tau, range(0,n0) );
        auto tau1 = slice( tau, range(n0,k) );
        
        // in this case T10 will be nonzero, (k-n0)-by-n0
        auto T10 = slice( TTT, range(n0,k), range(0,n0) );

        

        // two recursive calls on left half and right half
        larft_recursive( direction ,storeMode, Av0, tau0, T00);
        larft_recursive( direction ,storeMode,Av1, tau1, T11);


        // Step 1 ---- (V01)^H -> T01
        //auto A11 = slice(A, range(m-k, m-k+n0), range(n0, k));
        
        auto A11 = slice(A,range(n0, k),range(n-k, n-k+n0));
        
        //auto T10 = slice( TTT, range(n0,k), range(0,n0) );
        for (idx_t i = 0; i < k-n0; ++i) {
            for (idx_t j = 0; j < n0; ++j)
                T10(i,j) = A11(i,j);
        }

        // step 2--- T10 * V10 -> T10
        //auto A01 = slice(A, range(m-k, m-k+n0), range(0, n0));
        auto A01 = slice(A, range(0, n0),range(n-k, n-k+n0) );
        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Lower, tlapack::Op::Trans, tlapack::Diag::Unit, real_t(1), A01, T10 );

        if(n>m){
            auto A00 = slice(A,  range(0, n0),range(0, n-k));
            auto A10 = slice(A, range(n0, k),range(0, n-k));
            
            // auto A20 = slice(A, range(n, m), range(0, n0));
            // auto A21 = slice(A, range(n, m), range(n0, n));
            tlapack::gemm( tlapack:: Op::NoTrans, tlapack:: Op::ConjTrans , T(1),  A10,A00, T(1), T10 );
        }

        //step 4 ---  T11 * (- T10) -> T10
        tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Lower, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, real_t(-1), T11, T10 );

        //step 4 ---   (T10) * T00 -> T10
        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Lower, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, real_t(1), T00, T10 );

    }
    return 0;    
    

}


}//end tlapack

#endif // TLAPACK_LARFT_RECURSIVE_HH



