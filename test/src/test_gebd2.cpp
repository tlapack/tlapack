#include <utility>
#include <plugins/tlapack_stdvector.hpp>
#include <plugins/tlapack_legacyArray.hpp>
#include <plugins/tlapack_debugutils.hpp>
#include <tlapack.hpp>


#include <cstdlib> 
#include <chrono>  
#include <cstring>
#include "gebd2.h"

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

    // Aliases
    using T = std::complex<double>;
    // using T = float;
    // using T = double;
    using idx_t = size_t;
    using real_t = tlapack::real_type<T>;
    using range = std::pair<idx_t, idx_t>;
    using tlapack::conj;
    using tlapack::real;
    
    idx_t m = 7, n = 5, nb = 2;
    bool verbose  = false;

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
        if( strcmp( *(argv + i), "--verbose") == 0) {
			verbose  = true;
		}
	}

    if (n > m){
    // later to fix
        std::cout << "n > m, not supported (yet).";
        std::cout << std::endl;
        return 1;
    }

    std::cout << "m = " << m << "; " ;
    std::cout << "n = " << n << "; " ;
    std::cout << "nb = " << nb << "; "; 
    std::cout << std::endl;

    // Constants
    const double FLOPs = double(n) * double(n) * double(n) / 3;

    // set up our matrices
    std::vector<T> A_( m*n );
    tlapack::legacyMatrix<T> A( m, n, &A_[0], m ); 

    std::vector<T> B_( m*n );
    tlapack::legacyMatrix<T> B( m, n, &B_[0], m ); 

    std::vector<T> work(m); // masx of m and n

    std::vector<T> tauv(n); 
    std::vector<T> tauw(n); 

    
    //legacy matrix: column major format matrix, i loop in the inner of j look would be faster

    // Initialize a random seed
    srand( 3 );
    
    // Fill A with random entries
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i)
            A(i,j) = tlapack::make_scalar<T>(
                        rand_between_zero_and_one() + n, 
                        rand_between_zero_and_one() + n);
    }

    std::cout << std::setprecision(2);

    std::vector<T> C_( m*n );
    tlapack::legacyMatrix<T> C( m, n, &C_[0], m ); 
    tlapack::lacpy(tlapack::Uplo::General, A, C);

    gebd2(A, tauv, tauw, work);

    if ( verbose ) {
        std::cout << "A in terms of VW = [";
        tlapack::print_matrix( A );
        std::cout << "];" << std::endl;
        std::cout << std::endl;
        }

    for (idx_t j = 0; j < n; ++j) 
        for (idx_t i = 0; i < m; ++i)
	        B(i,j) = 0;

    B(0,0) = A(0,0);
    for (idx_t j = 1; j < n; ++j){
        B(j-1,j) = A(j-1,j); 
        B(j,j) = A(j,j); //get bidiagonal B
    }

    if(verbose){
        std::cout << "B = [";
        tlapack::print_matrix( B );
        std::cout << "];" << std::endl;
        std::cout << std::endl;
        }

    std::vector<T> D_( m*n );
    tlapack::legacyMatrix<T> D( m, n, &D_[0], m ); 

    tlapack::lacpy(tlapack::Uplo::General, B, D);
    
    
    //get Q from Householder reflector v's and Z from w's
    std::vector<T> Q_( m*m );
    tlapack::legacyMatrix<T> Q( m, m, &Q_[0], m ); 

    std::vector<T> Wq_( m*m );
    tlapack::legacyMatrix<T> Wq( m, m, &Wq_[0], m );

    //Identity matrix Q m-by-m to start with
    for (idx_t j = 0; j < m; ++j) {
        for (idx_t i = 0; i < m; ++i)
            Q(i,j) = tlapack::make_scalar<T>(0, 0);
        Q(j,j) = tlapack::make_scalar<T>(1, 0);
    }

    tlapack::lacpy(tlapack::Uplo::General, Q, Wq);

    tlapack::lacpy(tlapack::Uplo::Lower, A, Q);

    tlapack::ung2r(n, Q, tauv, work);

    // for (idx_t j = n-1; j != -1; --j) {
    //     auto v = tlapack::slice(A, range(j, m), j);
    //     auto Q11 = tlapack::slice(Q, range(j, m), range(0, m)); 
    //     tlapack::larf(tlapack::Side::Left, v, tauv[j], Q11, work);    
    // }

    tlapack::herk(tlapack::Uplo::Lower, tlapack::Op::NoTrans, real_t(1), Q, real_t(-1), Wq);
    real_t orth_Q = tlapack::lanhe(tlapack::Norm::Max, tlapack::Uplo::Lower, Wq);
    std::cout << "orth Q = " << orth_Q << "; "; 

    std::vector<T> Z_( n*n );
    tlapack::legacyMatrix<T> Z( n, n, &Z_[0], n ); 
    std::vector<T> Wz_( n*n );
    tlapack::legacyMatrix<T> Wz( n, n, &Wz_[0], n ); 

    //Identity matrix Z n-by-n to start with
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < n; ++i)
            Z(i,j) = tlapack::make_scalar<T>(0, 0);
        Z(j,j) = tlapack::make_scalar<T>(1, 0);
    }

    tlapack::lacpy(tlapack::Uplo::General, Z, Wz);

    for (idx_t j = n-2; j !=-1; --j) { 
        auto w = tlapack::slice(A, j, range(j+1, n));
        // for (idx_t i = 0; i < n-j-1; ++i) //for loop to conj w.
        //         w[i] = conj(w[i]);
        auto Z11 = tlapack::slice(Z, range(0, n), range(j+1, n));  
        
        tlapack::larf(tlapack::Side::Right, w, conj(tauw[j]), Z11, work); 
        // for (idx_t i = 0; i < n-j-1; ++i) //for loop to conj w back.
        //         w[i] = conj(w[i]);
    }

    tlapack::herk(tlapack::Uplo::Lower, tlapack::Op::NoTrans, real_t(1), Z, real_t(-1), Wz);
    real_t orth_Z = tlapack::lanhe(tlapack::Norm::Max, tlapack::Uplo::Lower, Wz);
    std::cout << "orth Z = " << orth_Z << "; "; 
    std::cout << std::endl;

    if (verbose) {
        std::cout << "Q = [";
        tlapack::print_matrix( Q );
        std::cout << "];" << std::endl;
        std::cout << std::endl;
    }

    if (verbose) {
        std::cout << "Z = [";
        tlapack::print_matrix( Z );
        std::cout << "];" << std::endl;
        std::cout << std::endl;
    }


    std::vector<T> K_( m*n );
    tlapack::legacyMatrix<T> K( m, n, &K_[0], m ); 
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i)
            K(i,j) = tlapack::make_scalar<T>(0, 0);
    }

    //2 ways to test
    //B = Q_trans * A * Z

    tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans, T(1), Q, C, T(0), K);

    if(verbose){
        std::cout << "K = [";
        tlapack::print_matrix( K );
        std::cout << "];" << std::endl;
        std::cout << std::endl;
        }

    real_t normB = tlapack::lange(tlapack::Norm::Max, B);


    tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans, T(1), K, Z, T(-1), B); //Trans for Z works for reals. ConjTrans not well for complex. this is wrong. should be NoTrans for Z
            //the Z we are getting is ConjTransed.
    if(verbose){
        std::cout << "B = [";
        tlapack::print_matrix( D );
        std::cout << "];" << std::endl;
        std::cout << std::endl;
        }

    real_t rel_error =tlapack::lange(tlapack::Norm::Max, B)/normB;
    std::cout << "error = " << rel_error << "; "<< std::endl;
    std::cout << std::endl;

    return 0;
    
    
}


    // for (idx_t j = 0; j < n; ++j) {
    //     auto a = tlapack::slice(A, range(j, m), j);
        
    //     tlapack::larfg(a, tau[j]);
    //     if( j < n-1 ){
    //         auto A11 = tlapack::slice(A, range(j, m), range(j+1, n));
    //         tlapack::larf(tlapack::Side::Left, a, conj(tau[j]), A11, work);
    //     }
        
    // }

    // std::vector<T> TT_( nb*n );
    // tlapack::legacyMatrix<T> TT( nb, n, &TT_[0], nb ); 

    // geqrf(A, TT, tau, work, nb); 

    // // tlapack::geqr2(A, tau, work);

    // // std::cout << tau[n]<< std::endl;



    // std::vector<T> Q_( m*m );
    // tlapack::legacyMatrix<T> Q( m, m, &Q_[0], m ); 

    // for (idx_t j = 0; j < n; ++j) {
    //     for (idx_t i = 0; i < m; ++i)
    //         Q(i,j) = tlapack::make_scalar<T>(
    //                     rand_between_zero_and_one(), 
    //                     rand_between_zero_and_one() );
    // }


    // // tlapack::lacpy(tlapack::Uplo::General, A, Q);

    // //  // Get an Identity matrix Q 
    // // for (idx_t j = 0; j < m; ++j) {
    // //     for (idx_t i = 0; i < m; ++i)
    // //         Q(i,j) = tlapack::make_scalar<T>(0, 0);
    // //     Q(j,j) = tlapack::make_scalar<T>(1, 0);
    // // }

    // // for (idx_t j = n-1; j != -1; --j) {
    // //     auto a = tlapack::slice(A, range(j, m), j);
    // //     auto Q11 = tlapack::slice(Q, range(j, m), range(0, m));
    // //     tlapack::larf(tlapack::Side::Left, a, tau[j], Q11, work);
    // // }

    // ungqr(A, Q, TT, tau, work, nb);

    // // tlapack::ung2r(n, Q, tau, work);

    // if( verbose ){
    //     std::cout << "Q = [";
    //     tlapack::print_matrix( Q );
    //     std::cout << "];" << std::endl;
    //     std::cout << std::endl;
    // }

    // std::vector<T> R_( m*n );
    // tlapack::legacyMatrix<T> R( m, n, &R_[0], m ); 

    // tlapack::lacpy(tlapack::Uplo::Upper, A, R);

    //  // Get R
    // for (idx_t j = 0; j < n; ++j) {
    //     for (idx_t i = j+1; i < m; ++i)
    //         R(i,j) = tlapack::make_scalar<T>(0, 0);
    // }

    // if( verbose ){
    //     std::cout << "R = [";
    //     tlapack::print_matrix( R );
    //     std::cout << "];" << std::endl;
    //     std::cout << std::endl;
    // }

    // std::vector<T> X_( m*m );
    // tlapack::legacyMatrix<T> X( m, m, &X_[0], m ); 
    // for (idx_t j = 0; j < m; ++j) {
    //     for (idx_t i = 0; i < j; ++i)
    //         X(i,j) = tlapack::make_scalar<T>(rand_between_zero_and_one(), 
    //         rand_between_zero_and_one());
    //     X(j,j) = tlapack::make_scalar<T>(1, 0);
    //     for (idx_t i = j+1; i < m; ++i)
    //         X(i,j) = tlapack::make_scalar<T>(0, 0);
    // }

    // tlapack::herk(tlapack::Uplo::Lower, tlapack::Op::ConjTrans, real_t(-1), Q, real_t(1), X);

    

    // real_t orth = tlapack::lanhe(tlapack::Norm::Max, tlapack::Uplo::Lower, X);
    // std::cout << "orth = " << orth << "; "; 


    // if( verbose ){
    //     std::cout << "X = [";
    //     tlapack::print_matrix( X );
    //     std::cout << "];" << std::endl;
    //     std::cout << std::endl;
    // }



    // // for (idx_t j = 0; j < n; ++j) {
        
    // //     auto a = tlapack::slice(A, range(j, m), j);
    // //     auto Q11 = tlapack::slice(Q, range(j, m), range(0, m));
    // //     tlapack::larf(tlapack::Side::Left, a, tau[j], Q11, work);
        
    // // 

    // real_t normA = tlapack::lange(tlapack::Norm::Max, B);
    // tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(-1), Q, R, T(1), B);
    // real_t repres = tlapack::lange(tlapack::Norm::Max, B)/normA;

    // std::cout << "repres = " << repres << "; "<< std::endl;

    

    // if( verbose ){
    //     std::cout << "B = [";
    //     tlapack::print_matrix( B );
    //     std::cout << "];" << std::endl;
    //     std::cout << std::endl;
    // }
