#include <utility>
#include <plugins/tlapack_stdvector.hpp>
#include <plugins/tlapack_legacyArray.hpp>
#include <plugins/tlapack_debugutils.hpp>
#include <tlapack.hpp>


#include <cstdlib> 
#include <chrono>  
#include <cstring>
#include "gelq2.h"
#include "ungl2.h"
#include "gelqf.h"


// Full k version using any 0 < k <= n. When k < m <= n, we are only using part of A to do LQ.

/** Returns a random number between 0 and 1 inclusive.
 * 
 * @return float 
 */
inline
float rand_between_zero_and_one() {
    return static_cast<float>( std::rand() )
            / static_cast<float>( RAND_MAX );
}

int main( int argc, char** argv ) {

    // using T = std::complex<double>;
    // using T = float;
    using T = double;
    using idx_t = size_t;
    using real_t = tlapack::real_type<T>;
    using range = std::pair<idx_t, idx_t>;
    using tlapack::conj;
    using tlapack::real;
    
    idx_t m = 5, n = 7, nb = 2, k = 3;
    bool verbose  = false;
    bool bidg  = false;

	for(idx_t i = 1; i < argc; i++){
        if( strcmp( *(argv + i), "-m") == 0) {
			m  = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-n") == 0) {
			n  = atoi( *(argv + i + 1) );
			i++;
		}
        if( strcmp( *(argv + i), "-nb") == 0) {
			nb  = atoi( *(argv + i + 1) );
			i++;
		}
        if( strcmp( *(argv + i), "-k") == 0) {
			k  = atoi( *(argv + i + 1) );
			i++;
		}
        if( strcmp( *(argv + i), "--bi") == 0) {
			bidg  = true;
		}
        if( strcmp( *(argv + i), "--verbose") == 0) {
			verbose  = true;
		}
	}

    std::cout << "m = " << std::setw(4) << m << "; " ;
    std::cout << "n = " << std::setw(4) << n << "; " ;
    std::cout << "k = " << std::setw(4) << k << "; " ;
    std::cout << "nb = " << std::setw(4) << nb << "; "; 
    // std::cout << std::endl;

    // Constants
    const double FLOPs = double(n) * double(n) * double(n) / 3;

    // set up our matrices
    std::vector<T> A_( m*n );
    tlapack::legacyMatrix<T> A( m, n, &A_[0], m ); 
    
    std::vector<T> B_( m*n );
    tlapack::legacyMatrix<T> B( m, n, &B_[0], m ); 

    std::vector<T> L_( k*n ); 
    tlapack::legacyMatrix<T> L( k, n, &L_[0], k ); 

    std::vector<T> TT_( m*nb );
    tlapack::legacyMatrix<T> TT( m, nb, &TT_[0], m ); 

    std::vector<T> work(std::max(m,n)); // masx of m and n

    std::vector<T> tauw(std::min(m,n)); //min of m and n.


    // Initialize a random seed
    srand( 3 );
    
    // Fill A with random entries
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i)
            A(i,j) = tlapack::make_scalar<T>(
                        rand_between_zero_and_one(), 
                        rand_between_zero_and_one());
    }

    std::cout << std::setprecision(2);

    tlapack::lacpy(tlapack::Uplo::General, A, B);

    if( verbose ){
        std::cout << "A =  [";
        tlapack::print_matrix( A );
        std::cout << "];" << std::endl;
        std::cout << std::endl;
    }

    gelq2(A, tauw, work);
    // gelqf(A, TT, tauw, work, nb);

    tlapack::lacpy(tlapack::Uplo::General, tlapack::slice(A, range(0, std::min(m, k)), range (0, n)), L); 

    if( verbose ){
        std::cout << "A in terms of Lw's = [";
        tlapack::print_matrix( A );
        std::cout << "];" << std::endl;
        std::cout << std::endl;
    }

    ungl2(L, tauw, work, bidg);

    auto Q = tlapack::slice( L, range(0,k), range(0,n) );

    if( verbose ){
        std::cout << "Q = [";
        tlapack::print_matrix( Q );
        std::cout << "];" << std::endl;
        std::cout << std::endl;
    }

    //Generate an identity matrix Wq m-by-m to start with
    std::vector<T> Wq_( k*k );
    tlapack::legacyMatrix<T> Wq( k, k, &Wq_[0], k );
    
    for (idx_t j = 0; j < k; ++j) {
        for (idx_t i = 0; i < k; ++i)
            Wq(i,j) = tlapack::make_scalar<T>(0, 0);
        Wq(j,j) = tlapack::make_scalar<T>(1, 0);
    }

    //Testing orthogonality of Q
    tlapack::herk(tlapack::Uplo::Upper, tlapack::Op::NoTrans, real_t(1), Q, real_t(-1), Wq);
    real_t orth_Q = tlapack::lanhe(tlapack::Norm::Max, tlapack::Uplo::Lower, Wq);
    std::cout << "orth Q = " << std::setw(8) << orth_Q << "; "; 

    // set up our matrices
    std::vector<T> R_( std::min(k,m)*n );
    tlapack::legacyMatrix<T> R( std::min(k,m), n, &R_[0], std::min(k,m) ); 

    // new things for k's
    std::vector<T> LL_( std::min(k,m)*k );
    tlapack::legacyMatrix<T> LL( std::min(k,m), k, &LL_[0], std::min(k,m) ); 
    tlapack::lacpy(tlapack::Uplo::Lower, tlapack::slice(A, range(0, std::min(m, k)), range (0, k)), LL); 
    
    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(1), LL, Q, T(0), R);


    //Test A = L * Q
    real_t normA = tlapack::lange(tlapack::Norm::Max, tlapack::slice(B, range(0, std::min(k,m)), range(0, n)));

    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < std::min(m,k); ++i)
            B(i,j) -= R(i,j);
        if (k < m){
            for (idx_t i = k; i < m; ++i)
                B(i,j) = tlapack::make_scalar<T>(0, 0);
        }
    }  

    if( verbose ){
        std::cout << "L * Q - A = [";
        tlapack::print_matrix( B );
        std::cout << "];" << std::endl;
        std::cout << std::endl;
    }

    real_t repres = tlapack::lange(tlapack::Norm::Max, B);
    std::cout << "repres = " << std::setw(8) << repres << "; "; 
    std::cout << std::endl;

    return 0;
    
}
