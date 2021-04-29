#include <tblas.hpp>

#include <vector>
#include <stdio.h>

//------------------------------------------------------------------------------
template <typename T>
void run( int m, int n, int k )
{
    int lda = m;
    int ldb = n;
    int ldc = m;
    std::vector<T> A( lda*k, 1.0 );  // m-by-k
    std::vector<T> B( ldb*n, 2.0 );  // k-by-n
    std::vector<T> C( ldc*n, 3.0 );  // m-by-n

    for (int col = 0; col < k; ++col) A[col*lda+col] = 1.0;
    for (int col = 0; col < k; ++col)
        for (int lin = 0; lin < lda; ++lin)
            A[col*lda+lin] = 0.0;

    for (int col = 0; col < n; ++col) B[col*ldb+col] = 1.0;
    for (int col = 0; col < n; ++col)
        for (int lin = 0; lin < ldb; ++lin)
            B[col*ldb+lin] = 0.0;

    for (int col = 0; col < n; ++col) C[col*ldc+col] = 1.0;
    for (int col = 0; col < n; ++col)
        for (int lin = 0; lin < ldc; ++lin)
            C[col*ldc+lin] = 0.0;

    // C = -1.0*A*B + 1.0*C
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                m, n, k,
                -1.0, A.data(), lda,
                      B.data(), ldb,
                 1.0, C.data(), ldc );

    for (int lin = 0; lin < ldc; ++lin) {
        printf( "\n");
        for (int col = 0; col < n; ++col)
            printf( "%lf ", C[col*ldc+lin]);
    }
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
//    int m = 100, n = 200, k = 50;
//    printf( "run< float >( %d, %d, %d )\n", m, n, k );
//    run< float  >( m, n, k );

//    printf( "run< double >( %d, %d, %d )\n", m, n, k );
//    run< double >( m, n, k );

//    printf( "run< complex<float> >( %d, %d, %d )\n", m, n, k );
//    run< std::complex<float>  >( m, n, k );

//    printf( "run< complex<double> >( %d, %d, %d )\n", m, n, k );
//    run< std::complex<double> >( m, n, k );

    int m = 3, n = 3, k = 1;
    printf( "run< float >( %d, %d, %d )\n", m, n, k );
    run< float  >( m, n, k );

    return 0;
}
