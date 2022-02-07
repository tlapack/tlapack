// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_UTIL_HH
#define LAPACK_UTIL_HH

#include "blas.hh"
#include "lapack/utils.hpp"
#include "slate_api/lapack/types.hpp"

namespace lapack {

    using blas::Error;

    inline char sides2char( Sides sides )
    {
        return char(sides);
    }

    inline const char* sides2str( Sides sides )
    {
        switch (sides) {
            case Sides::Left:  return "left";
            case Sides::Right: return "right";
            case Sides::Both:  return "both";
        }
        return "?";
    }

    inline Sides char2sides( char sides )
    {
        sides = (char) toupper( sides );
        assert( sides == 'L' || sides == 'R' || sides == 'B' );
        return Sides( sides );
    }

    // -----------------------------------------------------------------------------

    inline char norm2char( lapack::Norm norm )
    {
        return char( norm );
    }

    inline lapack::Norm char2norm( char norm )
    {
        norm = char( toupper( norm ));
        if (norm == 'O')
            norm = '1';
        else if (norm == 'E')
            norm = 'F';
        blas_error_if( norm != '1' && norm != '2' && norm != 'I' &&
                        norm != 'F' && norm != 'M' );
        return lapack::Norm( norm );
    }

    inline const char* norm2str( lapack::Norm norm )
    {
        switch (norm) {
            case Norm::One: return "1";
            case Norm::Two: return "2";
            case Norm::Inf: return "inf";
            case Norm::Fro: return "fro";
            case Norm::Max: return "max";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // Job for computing eigenvectors and singular vectors
    // # needs custom map

    inline char job2char( lapack::Job job )
    {
        return char( job );
    }

    // custom maps
    // bbcsd, orcsd2by1
    inline char job_csd2char( lapack::Job job )
    {
        switch (job) {
            case lapack::Job::Vec:          return 'Y';  // orcsd
            case lapack::Job::UpdateVec:    return 'Y';  // bbcsd
            default: return char( job );
        }
    }

    // bdsdc, gghrd, hgeqz, hseqr, pteqr, stedc, steqr, tgsja, trexc, trsen
    inline char job_comp2char( lapack::Job job )
    {
        switch (job) {
            case lapack::Job::Vec:          return 'I';
            case lapack::Job::UpdateVec:    return 'V';
            default: return char( job );
        }
    }

    // tgsja
    inline char job_compu2char( lapack::Job job )
    {
        switch (job) {
            case lapack::Job::Vec:          return 'I';
            case lapack::Job::UpdateVec:    return 'U';
            default: return char( job );
        }
    }

    // tgsja
    inline char job_compq2char( lapack::Job job )
    {
        switch (job) {
            case lapack::Job::Vec:          return 'I';
            case lapack::Job::UpdateVec:    return 'Q';
            default: return char( job );
        }
    }

    // ggsvd3, ggsvp3
    inline char jobu2char( lapack::Job job )
    {
        switch (job) {
            case lapack::Job::Vec:          return 'U';
            default: return char( job );
        }
    }

    // ggsvd3, ggsvp3
    inline char jobq2char( lapack::Job job )
    {
        switch (job) {
            case lapack::Job::Vec:          return 'Q';
            default: return char( job );
        }
    }

    // gejsva
    inline char jobu_gejsv2char( lapack::Job job )
    {
        switch (job) {
            case lapack::Job::SomeVec:      return 'U';
            case lapack::Job::AllVec:       return 'F';
            default: return char( job );
        }
    }

    // gesvj
    inline char job_gesvj2char( lapack::Job job )
    {
        switch (job) {
            case lapack::Job::SomeVec:      return 'U';  // jobu
            case lapack::Job::SomeVecTol:   return 'C';  // jobu
            case lapack::Job::UpdateVec:    return 'U';  // jobv
            default: return char( job );
        }
    }

    inline lapack::Job char2job( char job )
    {
        job = char( toupper( job ));
        blas_error_if( job != 'N' && job != 'V' && job != 'U' &&
                        job != 'A' && job != 'S' && job != 'O' &&
                        job != 'P' && job != 'C' && job != 'J' &&
                        job != 'W' );
        return lapack::Job( job );
    }

    inline const char* job2str( lapack::Job job )
    {
        switch (job) {
            case lapack::Job::NoVec:        return "novec";
            case lapack::Job::Vec:          return "vec";
            case lapack::Job::UpdateVec:    return "update";

            case lapack::Job::AllVec:       return "all";
            case lapack::Job::SomeVec:      return "some";
            case lapack::Job::OverwriteVec: return "overwrite";

            case lapack::Job::CompactVec:   return "compact";
            case lapack::Job::SomeVecTol:   return "sometol";
            case lapack::Job::VecJacobi:    return "jacobi";
            case lapack::Job::Workspace:    return "work";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // hseqr

    inline char jobschur2char( lapack::JobSchur jobschur )
    {
        return char( jobschur );
    }

    inline lapack::JobSchur char2jobschur( char jobschur )
    {
        jobschur = char( toupper( jobschur ));
        blas_error_if( jobschur != 'E' && jobschur != 'S' );
        return lapack::JobSchur( jobschur );
    }

    inline const char* jobschur2str( lapack::JobSchur jobschur )
    {
        switch (jobschur) {
            case lapack::JobSchur::Eigenvalues: return "eigval";
            case lapack::JobSchur::Schur:       return "schur";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // gees
    // todo: generic yes/no

    inline char sort2char( lapack::Sort sort )
    {
        return char( sort );
    }

    inline lapack::Sort char2sort( char sort )
    {
        sort = char( toupper( sort ));
        blas_error_if( sort != 'N' && sort != 'S' );
        return lapack::Sort( sort );
    }

    inline const char* sort2str( lapack::Sort sort )
    {
        switch (sort) {
            case lapack::Sort::NotSorted: return "not-sorted";
            case lapack::Sort::Sorted:    return "sorted";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // syevx

    inline char range2char( lapack::Range range )
    {
        return char( range );
    }

    inline lapack::Range char2range( char range )
    {
        range = char( toupper( range ));
        blas_error_if( range != 'A' && range != 'V' && range != 'I' );
        return lapack::Range( range );
    }

    inline const char* range2str( lapack::Range range )
    {
        switch (range) {
            case lapack::Range::All:   return "all";
            case lapack::Range::Value: return "value";
            case lapack::Range::Index: return "index";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------

    inline char vect2char( lapack::Vect vect )
    {
        return char( vect );
    }

    inline lapack::Vect char2vect( char vect )
    {
        vect = char( toupper( vect ));
        blas_error_if( vect != 'Q' && vect != 'P' && vect != 'N' && vect != 'B' );
        return lapack::Vect( vect );
    }

    inline const char* vect2str( lapack::Vect vect )
    {
        switch (vect) {
            case lapack::Vect::P:    return "p";
            case lapack::Vect::Q:    return "q";
            case lapack::Vect::None: return "none";
            case lapack::Vect::Both: return "both";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // larfb

    inline char direction2char( lapack::Direction direction )
    {
        return char( direction );
    }

    inline lapack::Direction char2direction( char direction )
    {
        direction = char( toupper( direction ));
        blas_error_if( direction != 'F' && direction != 'B' );
        return lapack::Direction( direction );
    }

    inline const char* direction2str( lapack::Direction direction )
    {
        switch (direction) {
            case lapack::Direction::Forward:  return "forward";
            case lapack::Direction::Backward: return "backward";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // larfb

    inline char storev2char( lapack::StoreV storev )
    {
        return char( storev );
    }

    inline lapack::StoreV char2storev( char storev )
    {
        storev = char( toupper( storev ));
        blas_error_if( storev != 'C' && storev != 'R' );
        return lapack::StoreV( storev );
    }

    inline const char* storev2str( lapack::StoreV storev )
    {
        switch (storev) {
            case lapack::StoreV::Columnwise: return "columnwise";
            case lapack::StoreV::Rowwise:    return "rowwise";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // lascl, laset

    inline char matrixtype2char( lapack::MatrixType type )
    {
        return char( type );
    }

    inline lapack::MatrixType char2matrixtype( char type )
    {
        type = char( toupper( type ));
        blas_error_if( type != 'G' && type != 'L' && type != 'U' &&
                        type != 'H' && type != 'B' && type != 'Q' && type != 'Z' );
        return lapack::MatrixType( type );
    }

    inline const char* matrixtype2str( lapack::MatrixType type )
    {
        switch (type) {
            case lapack::MatrixType::General:    return "general";
            case lapack::MatrixType::Lower:      return "lower";
            case lapack::MatrixType::Upper:      return "upper";
            case lapack::MatrixType::Hessenberg: return "hessenberg";
            case lapack::MatrixType::LowerBand:  return "band-lower";
            case lapack::MatrixType::UpperBand:  return "q-band-upper";
            case lapack::MatrixType::Band:       return "z-band";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // trevc

    inline char howmany2char( lapack::HowMany howmany )
    {
        return char( howmany );
    }

    inline lapack::HowMany char2howmany( char howmany )
    {
        howmany = char( toupper( howmany ));
        blas_error_if( howmany != 'A' && howmany != 'B' && howmany != 'S' );
        return lapack::HowMany( howmany );
    }

    inline const char* howmany2str( lapack::HowMany howmany )
    {
        switch (howmany) {
            case lapack::HowMany::All:           return "all";
            case lapack::HowMany::Backtransform: return "backtransform";
            case lapack::HowMany::Select:        return "select";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // *svx, *rfsx

    inline char equed2char( lapack::Equed equed )
    {
        return char( equed );
    }

    inline lapack::Equed char2equed( char equed )
    {
        equed = char( toupper( equed ));
        blas_error_if( equed != 'N' && equed != 'R' && equed != 'C' &&
                        equed != 'B' && equed != 'Y' );
        return lapack::Equed( equed );
    }

    inline const char* equed2str( lapack::Equed equed )
    {
        switch (equed) {
            case lapack::Equed::None: return "none";
            case lapack::Equed::Row:  return "row";
            case lapack::Equed::Col:  return "col";
            case lapack::Equed::Both: return "both";
            case lapack::Equed::Yes:  return "yes";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // *svx
    // todo: what's good name for this?

    inline char factored2char( lapack::Factored factored )
    {
        return char( factored );
    }

    inline lapack::Factored char2factored( char factored )
    {
        factored = char( toupper( factored ));
        blas_error_if( factored != 'F' && factored != 'N' && factored != 'E' );
        return lapack::Factored( factored );
    }

    inline const char* factored2str( lapack::Factored factored )
    {
        switch (factored) {
            case lapack::Factored::Factored:    return "factored";
            case lapack::Factored::NotFactored: return "notfactored";
            case lapack::Factored::Equilibrate: return "equilibrate";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // geesx, trsen

    inline char sense2char( lapack::Sense sense )
    {
        return char( sense );
    }

    inline lapack::Sense char2sense( char sense )
    {
        sense = char( toupper( sense ));
        blas_error_if( sense != 'N' && sense != 'E' && sense != 'V' &&
                        sense != 'B' );
        return lapack::Sense( sense );
    }

    inline const char* sense2str( lapack::Sense sense )
    {
        switch (sense) {
            case lapack::Sense::None:        return "none";
            case lapack::Sense::Eigenvalues: return "eigval";
            case lapack::Sense::Subspace:    return "subspace";
            case lapack::Sense::Both:        return "both";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // disna

    inline char jobcond2char( lapack::JobCond jobcond )
    {
        return char( jobcond );
    }

    inline lapack::JobCond char2jobcond( char jobcond )
    {
        jobcond = char( toupper( jobcond ));
        blas_error_if( jobcond != 'N' && jobcond != 'E' && jobcond != 'V' &&
                        jobcond != 'B' );
        return lapack::JobCond( jobcond );
    }

    inline const char* jobcond2str( lapack::JobCond jobcond )
    {
        switch (jobcond) {
            case lapack::JobCond::EigenVec:         return "eigvec";
            case lapack::JobCond::LeftSingularVec:  return "left";
            case lapack::JobCond::RightSingularVec: return "right";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // {ge,gg}{bak,bal}

    inline char balance2char( lapack::Balance balance )
    {
        return char( balance );
    }

    inline lapack::Balance char2balance( char balance )
    {
        balance = char( toupper( balance ));
        blas_error_if( balance != 'N' && balance != 'P' && balance != 'S' &&
                        balance != 'B' );
        return lapack::Balance( balance );
    }

    inline const char* balance2str( lapack::Balance balance )
    {
        switch (balance) {
            case lapack::Balance::None:    return "none";
            case lapack::Balance::Permute: return "permute";
            case lapack::Balance::Scale:   return "scale";
            case lapack::Balance::Both:    return "both";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // stebz, larrd, stein docs

    inline char order2char( lapack::Order order )
    {
        return char( order );
    }

    inline lapack::Order char2order( char order )
    {
        order = char( toupper( order ));
        blas_error_if( order != 'B' && order != 'E' );
        return lapack::Order( order );
    }

    inline const char* order2str( lapack::Order order )
    {
        switch (order) {
            case lapack::Order::Block:  return "block";
            case lapack::Order::Entire: return "entire";
        }
        return "?";
    }

    // -----------------------------------------------------------------------------
    // check_ortho (LAPACK testing zunt01)

    inline char rowcol2char( lapack::RowCol rowcol )
    {
        return char( rowcol );
    }

    inline lapack::RowCol char2rowcol( char rowcol )
    {
        rowcol = char( toupper( rowcol ));
        blas_error_if( rowcol != 'C' && rowcol != 'R' );
        return lapack::RowCol( rowcol );
    }

    inline const char* rowcol2str( lapack::RowCol rowcol )
    {
        switch (rowcol) {
            case lapack::RowCol::Col: return "col";
            case lapack::RowCol::Row: return "row";
        }
        return "?";
    }

}

#endif  // LAPACK_UTIL_HH
