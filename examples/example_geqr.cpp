#include <tlapack.hpp>

#include <vector>
#include <stdio.h>
#include <stdlib.h> 
#include <time.h> 

//------------------------------------------------------------------------------
template <typename T>
void run( blas::size_t m, blas::size_t n )
{
    blas::int_t lda = std::max<blas::int_t>(m,1);
    blas::size_t tsize = 100;

    std::vector<T> A( lda*n );  // m-by-n
    std::vector<T> t( tsize );

    for (blas::size_t j = 0; j < n; ++j)
        for (blas::size_t i = 0; i < lda; ++i)
            A[j*lda+i] = static_cast<float>( rand() )
                        / static_cast<float>( RAND_MAX );

    lapack::geqr( m, n, A.data(), lda, t.data(), tsize );

    for (blas::size_t i = 0; i < lda; ++i) {
        printf( "\n");
        for (blas::size_t j = 0; j < n; ++j)
            printf( "%lf ", blas::real(A[j*lda+i]) );
    }
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    int m = 10;
    int n = 10;
    srand( (unsigned)time(NULL) );
    
    printf( "\n\nrun< float  >( %d, %d )", m, n );
    run< float  >( m, n );
    
    printf( "\n\nrun< double >( %d, %d )", m, n );
    run< double >( m, n );
    
    printf( "\n\nrun< std::complex<float>  >( %d, %d )", m, n );
    run< std::complex<float>  >( m, n );
    
    printf( "\n\nrun< std::complex<double> >( %d, %d )", m, n );
    run< std::complex<double> >( m, n );

    return 0;
}
