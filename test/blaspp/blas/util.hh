/// @file util.hh
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_UTIL_HH
#define BLAS_UTIL_HH

#include "tlapack/base/utils.hpp"

/// Use to silence compiler warning of unused variable.
#define blas_unused( var ) ((void)var)

#define blas_error_if( cond ) tlapack_check_false( cond )

namespace blas {

    using tlapack::real_type;
    using tlapack::complex_type;
    using tlapack::scalar_type;

    using tlapack::Layout;
    using tlapack::Op;
    using tlapack::Uplo;
    using tlapack::Diag;
    using tlapack::Side;

    using tlapack::real;
    using tlapack::imag;
    using tlapack::conj;

    //------------------------------------------------------------------------------
    /// True if T is std::complex<T2> for some type T2.
    template <typename T>
    struct is_complex:
        std::integral_constant<bool, false>
    {};

    // specialize for std::complex
    template <typename T>
    struct is_complex< std::complex<T> >:
        std::integral_constant<bool, true>
    {};

    // Empty structure since <T>LAPACK is not defining device BLAS
    struct Queue
    {
        void sync() {}
    };

    class Error: public std::exception {
    public:
        /// Constructs BLAS error
        Error():
            std::exception()
        {}

        /// Constructs BLAS error with message
        Error( std::string const& msg ):
            std::exception(),
            msg_( msg )
        {}

        /// Constructs BLAS error with message: "msg, in function func"
        Error( const char* msg, const char* func ):
            std::exception(),
            msg_( std::string(msg) + ", in function " + func )
        {}

        /// Returns BLAS error message
        virtual const char* what() const noexcept override
            { return msg_.c_str(); }

    private:
        std::string msg_;
    };

    // -----------------------------------------------------------------------------
    // New enum
    enum class Format : char { LAPACK   = 'L', Tile     = 'T' };

    // -----------------------------------------------------------------------------
    // Convert enum to LAPACK-style char.
    inline char layout2char( Layout layout ) { return char(layout); }
    inline char     op2char( Op     op     ) { return char(op);     }
    inline char   uplo2char( Uplo   uplo   ) { return char(uplo);   }
    inline char   diag2char( Diag   diag   ) { return char(diag);   }
    inline char   side2char( Side   side   ) { return char(side);   }
    inline char format2char( Format format ) { return char(format); }

    // -----------------------------------------------------------------------------
    // Convert enum to LAPACK-style string.
    inline const char* layout2str( Layout layout )
    {
        switch (layout) {
            case Layout::ColMajor: return "col";
            case Layout::RowMajor: return "row";
            default:               return "";
        }
    }

    inline const char* op2str( Op op )
    {
        switch (op) {
            case Op::NoTrans:   return "notrans";
            case Op::Trans:     return "trans";
            case Op::ConjTrans: return "conj";
            default:            return "";
        }
    }

    inline const char* uplo2str( Uplo uplo )
    {
        switch (uplo) {
            case Uplo::Lower:   return "lower";
            case Uplo::Upper:   return "upper";
            case Uplo::General: return "general";
            default:            return "";
        }
    }

    inline const char* diag2str( Diag diag )
    {
        switch (diag) {
            case Diag::NonUnit: return "nonunit";
            case Diag::Unit:    return "unit";
        }
        return "";
    }

    inline const char* side2str( Side side )
    {
        switch (side) {
            case Side::Left:  return "left";
            case Side::Right: return "right";
        }
        return "";
    }

    inline const char* format2str( Format format )
    {
        switch (format) {
            case Format::LAPACK: return "lapack";
            case Format::Tile: return "tile";
        }
        return "";
    }

    // -----------------------------------------------------------------------------
    // Convert LAPACK-style char to enum.
    inline Layout char2layout( char layout )
    {
        layout = (char) toupper( layout );
        assert( layout == 'C' || layout == 'R' );
        return Layout( layout );
    }

    inline Op char2op( char op )
    {
        op = (char) toupper( op );
        assert( op == 'N' || op == 'T' || op == 'C' );
        return Op( op );
    }

    inline Uplo char2uplo( char uplo )
    {
        uplo = (char) toupper( uplo );
        assert( uplo == 'L' || uplo == 'U' || uplo == 'G' );
        return Uplo( uplo );
    }

    inline Diag char2diag( char diag )
    {
        diag = (char) toupper( diag );
        assert( diag == 'N' || diag == 'U' );
        return Diag( diag );
    }

    inline Side char2side( char side )
    {
        side = (char) toupper( side );
        assert( side == 'L' || side == 'R' );
        return Side( side );
    }

    inline Format char2format( char format )
    {
        format = (char) toupper( format );
        assert( format == 'L' || format == 'T' );
        return Format( format );
    }

    // max
    template <typename T1, typename T2>
    inline scalar_type<T1, T2> max(const T1& x, const T2& y)
    {
        return (x >= y ? x : y);
    }
    template <typename T1, typename T2, typename... Types>
    inline scalar_type<T1, T2, Types...> max(const T1& first, const T2& second, const Types&... args)
    {
        return max(first, max(second, args...));
    }

    // min
    template <typename T1, typename T2>
    inline scalar_type<T1, T2> min(const T1& x, const T2& y)
    {
        return (x <= y ? x : y);
    }
    template <typename T1, typename T2, typename... Types>
    inline scalar_type<T1, T2, Types...> min(const T1& first, const T2& second, const Types&... args)
    {
        return min(first, min(second, args...));
    }
}

using blas::uplo2char;
using blas::side2char;
using blas::op2char;
using blas::layout2char;
using blas::diag2char;

using blas::uplo2str;
using blas::side2str;
using blas::op2str;
using blas::layout2str;
using blas::diag2str;

#endif
