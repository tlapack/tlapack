#ifndef __DEFINES_H__
#define __DEFINES_H__

#include <stdint.h>

// -----------------------------------------------------------------------------
#include <complex.h>
typedef float _Complex  complexFloat;
typedef double _Complex complexDouble;

// -----------------------------------------------------------------------------
typedef enum { ColMajor = 'C', RowMajor = 'R' } Layout;
typedef enum { NoTrans  = 'N', Trans    = 'T', ConjTrans = 'C' } Op;
typedef enum { Upper    = 'U', Lower    = 'L', General   = 'G' } Uplo;
typedef enum { NonUnit  = 'N', Unit     = 'U' } Diag;
typedef enum { Left     = 'L', Right    = 'R' } Side;

#endif // __DEFINES_H__