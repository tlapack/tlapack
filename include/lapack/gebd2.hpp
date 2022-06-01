#ifndef GEBD2
#define GEBD2

#include <utility>
#include <plugins/tlapack_stdvector.hpp>
#include <plugins/tlapack_legacyArray.hpp>
#include <plugins/tlapack_debugutils.hpp>
#include <tlapack.hpp>

/** 
* hand-crafted  reduction to bidiagonal form of a general m-by-n matrix A, level 2, right looking algorithm
* @param[in,out] A
*/

template <typename matrix_t, class vector_t>
int gebd2( matrix_t & A, vector_t & tauv, vector_t & tauw, vector_t & work ){

    using idx_t = tlapack::size_type< matrix_t >;
    using T = tlapack::type_t< matrix_t >;
    using tlapack::conj;
    using tlapack::real;
    using real_t = tlapack::real_type<T>;
    using range = std::pair<idx_t, idx_t>;


    const idx_t m = tlapack::nrows( A );
    const idx_t n = tlapack::ncols( A );

    for (idx_t j = 0; j < n; ++j) {

        auto v = tlapack::slice(A, range(j, m), j);
        
        tlapack::larfg(v, tauv[j]); //gen the vertical reflector v's

        if( j < n-1 ){ 
            auto A11 = tlapack::slice(A, range(j, m), range(j+1, n));
            tlapack::larf(tlapack::Side::Left, v, conj(tauv[j]), A11, work);
        }

        if( j < n-1 ){
            auto w = tlapack::slice(A, j, range(j+1, n)); 
            //for loop to conj w.
            for (idx_t i = 0; i < n-j-1; ++i)
                w[i] = conj(w[i]);

            tlapack::larfg(w, tauw[j]); //gen the horizontal reflector w's

            if( j < m-1 ){ 
                auto B11 = tlapack::slice(A, range(j+1, m), range(j+1, n));
                tlapack::larf(tlapack::Side::Right, w, tauw[j], B11, work);
            }
            // for (idx_t i = 0; i < n-j-1; ++i) //for loop to conj w back.
            //     w[i] = conj(w[i]);
        }
    }

    return 0;

    // Handwritten Q*B, but it runs into dimension error. 
    // for (idx_t i = 0; i < m; ++i)
    //     Q(i,0) = Q(i,0) * B(0,0);
    // for (idx_t j = 1; j < n; ++j){
    //     for (idx_t i = 0; i < m; ++i)
    //         Q(i,j) = Q(i,j-1)* B(j-1,j) + Q(i,j)*B(j,j);
    // }

    // std::cout << "Q*B = [";
    // tlapack::print_matrix( Q );
    // std::cout << "];" << std::endl;
    // std::cout << std::endl;



    // 2 tests for the factorization.
    //1) B = Q_trans * A * Z. This test is in the main.cpp

    //2) A = Q * B * Z_trans
    // tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(1), Q, B, T(0), K);

    // std::cout << "K = [";
    // tlapack::print_matrix( K );
    // std::cout << "];" << std::endl;
    // std::cout << std::endl;

    // real_t normA = tlapack::lange(tlapack::Norm::Max, C);

    // tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans, T(-1), K, Z, T(1), C);


    // real_t rel_error =tlapack::lange(tlapack::Norm::Max, C)/normA;
    // std::cout << "error = " << rel_error << "; "<< std::endl;


}
#endif //GEBD2