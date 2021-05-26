#!/usr/bin/env python3
## @brief Create and populate 'test_corner_cases.cpp'
#
#  @author Weslley S. Pereira, University of Colorado Denver
#  @date   March 30, 2021
#
# Copyright (c) 2021, University of Colorado Denver. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# ------------------------------------------------------------------------------
# Test if s can be converted into an integer
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

# ------------------------------------------------------------------------------
# Test if s can be converted into a Floating point number
def RepresentsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

# ------------------------------------------------------------------------------
# Size of the arrays in the tests:
sizeArr = 5
sizeArray = str(sizeArr)

# ------------------------------------------------------------------------------
# BLAS routines' definitions
blas_routines = open("blas_routines.csv").read().splitlines()
for i in range(len(blas_routines)):
    blas_routines[i] = list(filter(None, [x.strip() for x in blas_routines[i].split(',')]))

# Functions that only accepts real types
realOnly_funcs = ["rotm", "rotmg", "syr", "symv"]
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# General rules that shall throw an Exception
throwException_corner_rules = {}
with open("throwException_corner_rules.csv") as f:
    for line in f:
        l = list(filter(None, [x.strip() for x in line.split(',')]))
        throwException_corner_rules[ int(l[0]) ] = l[1:]
# Tests that shall throw an Exception
throwException_corner_tests = {}
with open("throwException_corner_tests.csv") as f:
    for line in f:
        l = list(filter(None, [x.strip() for x in line.split(',')]))
        throwException_corner_tests[ l[0] ] = l[1:]

# ------------------------------------------------------------------------------
# Rules that result in invalid configurations which
# shall not throw any Exception nor modify the arguments
returnImmediately_corner_rules = {}
with open("returnImmediately_corner_rules.csv") as f:
    for line in f:
        l = list(filter(None, [x.strip() for x in line.split(',')]))
        returnImmediately_corner_rules[ int(l[0]) ] = l[1:]
# Tests that shall throw no Exception nor modify the arguments
returnImmediately_corner_tests = {}
with open("returnImmediately_corner_tests.csv") as f:
    for line in f:
        l = list(filter(None, [x.strip() for x in line.split(',')]))
        returnImmediately_corner_tests[ l[0] ] = l[1:]

# ------------------------------------------------------------------------------
# Print header of the test file:
print("""\
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <type_traits>
#include <catch2/catch.hpp>
#include <tblas.hpp>
#include "test_types.hpp"

using namespace blas;""")

# ------------------------------------------------------------------------------
# Loop in the functions
for idx, line in enumerate(blas_routines):
    f_name = line[0]
    blas_lv = int(line[1])
    ref_types = line[2::2]
    ref_args = line[3::2]

    # --------------------------------------------------------------------------
    # Print the parameters
    buffer = """\
TEMPLATE_TEST_CASE( \"""" + f_name + """ satisfies all corner cases", "[""" + \
        f_name + """][BLASlv""" + str(blas_lv) + """]", """ + \
        ("TEST_TYPES" if f_name not in realOnly_funcs else "TEST_REAL_TYPES") + \
        """ ) {
    using real_t = real_type<TestType>;
    
    // Default arguments:""",
    for j, arg in enumerate(ref_args):
        isArray = True if "*" in ref_types[j] else False
        if not isArray:
            buffer += """
    """ + ref_types[j] + " " + arg,
        else:
            buffer += """
    """ + ref_types[j].replace("*", "") + arg + "[]",
        if "Layout" in ref_types[j]:
            buffer += " = Layout::ColMajor;",
        elif "Op" in ref_types[j]:
            buffer += " = Op::NoTrans;",
        elif "Uplo" in ref_types[j]:
            buffer += " = Uplo::Upper;",
        elif "Diag" in ref_types[j]:
            buffer += " = Diag::NonUnit;",
        elif "Side" in ref_types[j]:
            buffer += " = Side::Left;",
        elif isArray and "TestType" in ref_types[j]:
            buffer += " = {",
            for k in range(sizeArr-1):
                buffer += "real_t(1), ",
            buffer += "real_t(1)};",
        elif "TestType" in ref_types[j]:
            buffer += " = real_t(1);",
        else:
            buffer += " = 1;",
            
    buffer += """

    // Corner cases:""",
    countCases = 0

    # --------------------------------------------------------------------------
    # Tests that throw an Error Exception:
    throwExceptionBuffer = ()
    for k in throwException_corner_tests[f_name]:
        protect_sizet = False
        try:
            if RepresentsInt(k):
                configStr = throwException_corner_rules[int(k)][0]
            else:
                configStr = k
            attribs = [x.strip() for x in configStr.split(';')]
            args = ref_args.copy()
            for varAttrib in attribs:
                param, value = [x.strip() for x in varAttrib.split('=', 1)]
                for i, x in enumerate(args):
                    if x == param:
                        if RepresentsFloat(value) and ("size_t" in ref_types[i]):
                            if float(value) < 0:
                                protect_sizet = True
                        args[i] = value
                        break
        except: # Invalid test
            continue
        if protect_sizet:
            throwExceptionBuffer += """
        if( std::is_signed<blas::size_t>::value )
            CHECK_THROWS_AS( """ + f_name + "( " + ", ".join(args) + " ), Error );",
        else:
            throwExceptionBuffer += """
        CHECK_THROWS_AS( """ + f_name + "( " + ", ".join(args) + " ), Error );",
        countCases += 1

    if countCases > 0:
        buffer += """
    SECTION( "Throw Error Tests" ) {""" + "".join(throwExceptionBuffer) + """
    }""",

    # Tests that return imediately:
    countCasesNoChange = 0
    refVarBuffer = ()
    swapBuffer = ()
    noChangeBuffer = ()
    bufferRequireNoChanges = ()
    for j, arg in enumerate(ref_args):
        if "*" in ref_types[j] and "const" not in ref_types[j]:
            refVarBuffer += """
        """ + ref_types[j].replace("*", "") + "ref_"+arg + "[" + sizeArray + """];
        std::copy( """ + arg + ", " + arg+"+"+sizeArray + ", ref_"+arg + " );",
            noChangeBuffer += "std::equal("+arg+", "+arg+"+"+sizeArray+", ref_"+arg+")",
            swapBuffer += """
        std::swap( """ + arg + ", ref_" + arg + " );",
    noChangeStr = "("+" && ".join(noChangeBuffer)+")"
    refVarStr = "".join(refVarBuffer)
    swapStr = "".join(swapBuffer)

    for k in returnImmediately_corner_tests[f_name]:
        protect_sizet = False
        try:
            if RepresentsInt(k):
                configStr = returnImmediately_corner_rules[int(k)][0]
            else:
                configStr = k
            attribs = [x.strip() for x in configStr.split(';')]
            args = ref_args.copy()
            for varAttrib in attribs:
                param, value = [x.strip() for x in varAttrib.split('=', 1)]
                for i, x in enumerate(args):
                    if x == param:
                        if RepresentsFloat(value) and ("size_t" in ref_types[i]):
                            if float(value) < 0:
                                protect_sizet = True
                        args[i] = value
                        break
        except: # Invalid test
            continue

        if protect_sizet:
            bufferRequireNoChanges += """
    if( std::is_signed<blas::size_t>::value ) {
    SECTION( \"""" + configStr + """\" ) {""" + refVarStr + """
        REQUIRE_NOTHROW( """ + f_name + "( " + ", ".join(args) + """ ) );
        CHECK( """ + noChangeStr + """ );""" + swapStr + """
    }}""",
        else:
            bufferRequireNoChanges += """
    SECTION( \"""" + configStr + """\" ) {""" + refVarStr + """
        REQUIRE_NOTHROW( """ + f_name + "( " + ", ".join(args) + """ ) );
        CHECK( """ + noChangeStr + """ );""" + swapStr + """
    }""",
        countCasesNoChange += 1

    if countCasesNoChange > 0:
        buffer += bufferRequireNoChanges

    # Specific tests Lv1:
    if f_name == "rotmg":
        buffer += """
    SECTION ( "Throw if d1 == -1" ) {
        real_t d1Minus1 = real_t(-1);
        CHECK_THROWS_AS( rotmg( &d1Minus1, d2, a, b, param ), Error );
    }""",
        countCases += 1
    elif f_name == "dot" or f_name == "dotu":
        buffer += """
    SECTION ( "n <= 0" ) {
        if( std::is_signed<blas::size_t>::value )
            CHECK( """+f_name+"""(-1, x, incx, y, incy ) == real_t(0) );
        CHECK( """+f_name+"""( 0, x, incx, y, incy ) == real_t(0) );
    }""",
        countCases += 2
    elif f_name == "asum" or f_name == "nrm2":
        buffer += """
    SECTION ( "n <= 0" ) {
        if( std::is_signed<blas::size_t>::value )
            CHECK( """+f_name+"""(-1, x, incx ) == real_t(0) );
        CHECK( """+f_name+"""( 0, x, incx ) == real_t(0) );
    }""",
        countCases += 2
    elif f_name == "iamax":
        buffer += """
    SECTION ( "n <= 0" ) {
        if( std::is_signed<blas::size_t>::value )
            CHECK( iamax(-1, x, incx ) == INVALID_INDEX );
        CHECK( iamax( 0, x, incx ) == INVALID_INDEX );
    }""",
        countCases += 2

    # Specific tests Lv2:
    elif f_name == "gemv":
        buffer += """
    SECTION( "y does not need to be set if beta = 0" ) {
        y[0] = y[1] = real_t(NAN);
        REQUIRE_NOTHROW( gemv( layout, trans, 2, 2, alpha, A, 2, x, incx, real_t(0), y, incy ) );
        CHECK( (y[0] == y[0] && y[1] == y[1]) ); // i.e., they are not NaN
        y[0] = y[1] = 1;
    }""",
        countCases += 1
    elif f_name == "hemv":
        buffer += """
    SECTION( "y does not need to be set if beta = 0" ) {
        y[0] = y[1] = real_t(NAN);
        REQUIRE_NOTHROW( hemv( layout, uplo, 2, alpha, A, 2, x, incx, real_t(0), y, incy ) );
        CHECK( (y[0] == y[0] && y[1] == y[1]) ); // i.e., they are not NaN
        y[0] = y[1] = 1;
    }
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of A is not referenced" ) {
        complex_t const _A[] = {{1, real_t(NAN)}, real_t(1), real_t(1), {1, real_t(NAN)}};
        REQUIRE_NOTHROW( hemv( layout, uplo, 2, alpha, _A, 2, (complex_t const *)x, incx, beta, (complex_t *)y, incy ) );
        CHECK( (y[0] == y[0] && y[1] == y[1]) ); // i.e., they are not NaN
        y[0] = y[1] = 1;
    }}""",
        countCases += 2
    elif f_name == "symv":
        buffer += """
    SECTION( "y does not need to be set if beta = 0" ) {
        y[0] = y[1] = real_t(NAN);
        REQUIRE_NOTHROW( symv( layout, uplo, 2, alpha, A, 2, x, incx, real_t(0), y, incy ) );
        CHECK( (y[0] == y[0] && y[1] == y[1]) ); // i.e., they are not NaN
        y[0] = y[1] = 1;
    }""",
        countCases += 1
    elif f_name == "her":
        buffer += """
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of A is zero" ) {
        complex_t _A[] = {{1, real_t(NAN)}, real_t(1), real_t(1), {1, real_t(NAN)}};
        REQUIRE_NOTHROW( her( layout, uplo, 2, alpha, (complex_t const *)x, incx, _A, 2 ) );
        CHECK( (_A[0] == _A[0] && _A[1] == _A[1] && _A[2] == _A[2] && _A[3] == _A[3]) ); // i.e., they are not NaN
        CHECK( (std::imag(_A[0]) == real_t(0) && std::imag(_A[3]) == real_t(0)) );
    }}""",
        countCases += 2
    elif f_name == "her2":
        buffer += """
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of A is zero" ) {
        complex_t _A[] = {{1, real_t(NAN)}, real_t(1), real_t(1), {1, real_t(NAN)}};
        REQUIRE_NOTHROW( her2( layout, uplo, 2, alpha, (complex_t const *)x, incx, y, incy, _A, 2 ) );
        CHECK( (_A[0] == _A[0] && _A[1] == _A[1] && _A[2] == _A[2] && _A[3] == _A[3]) ); // i.e., they are not NaN
        CHECK( (std::imag(_A[0]) == real_t(0) && std::imag(_A[3]) == real_t(0)) );
    }}""",
        countCases += 2
    elif f_name == "trmv" or f_name == "trsv":
        buffer += """
    SECTION( "Diagonal of A is not referenced if diag = 'U'" ) {
        TestType const _A[] = {real_t(NAN), real_t(1), real_t(1), real_t(NAN)};
        REQUIRE_NOTHROW( """+f_name+"""( layout, uplo, trans, Diag('U'), 2, _A, 2, x, incx ) );
        CHECK( (x[0] == x[0] && x[1] == x[1]) ); // i.e., they are not NaN
        x[0] = x[1] = 1;
    }""",
        countCases += 1

    # Specific tests Lv3:
    elif f_name == "gemm":
        buffer += """
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = real_t(2)*C[0];
        REQUIRE_NOTHROW( gemm( layout, transA, transB, m, n, 0, alpha, A, lda, B, ldb, real_t(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }""",
        countCases += 1
    elif f_name == "syrk":
        buffer += """
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = real_t(2)*C[0];
        REQUIRE_NOTHROW( """+f_name+"""( layout, uplo, trans, n, 0, alpha, A, lda, real_t(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }""",
        countCases += 1
    elif f_name == "syr2k":
        buffer += """
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = real_t(2)*C[0];
        REQUIRE_NOTHROW( """+f_name+"""( layout, uplo, trans, n, 0, alpha, A, lda, B, ldb, real_t(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }""",
        countCases += 1
    elif f_name == "herk":
        buffer += """
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = real_t(2)*C[0];
        REQUIRE_NOTHROW( """+f_name+"""( layout, uplo, trans, n, 0, alpha, A, lda, real_t(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of C is zero" ) {
        complex_t _C[] = {{1, real_t(NAN)}, real_t(1), real_t(1), {1, real_t(NAN)}};
        REQUIRE_NOTHROW( herk( layout, uplo, trans, 2, 2, alpha, A, 2, beta, _C, 2 ) );
        CHECK( (_C[0] == _C[0] && _C[1] == _C[1] && _C[2] == _C[2] && _C[3] == _C[3]) ); // i.e., they are not NaN
    }}""",
        countCases += 2
    elif f_name == "her2k":
        buffer += """
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = real_t(2)*C[0];
        REQUIRE_NOTHROW( """+f_name+"""( layout, uplo, trans, n, 0, alpha, A, lda, B, ldb, real_t(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of C is zero" ) {
        complex_t _C[] = {{1, real_t(NAN)}, real_t(1), real_t(1), {1, real_t(NAN)}};
        REQUIRE_NOTHROW( her2k( layout, uplo, trans, 2, 2, alpha, A, 2, B, 2, beta, _C, 2 ) );
        CHECK( (_C[0] == _C[0] && _C[1] == _C[1] && _C[2] == _C[2] && _C[3] == _C[3]) ); // i.e., they are not NaN
    }}""",
        countCases += 2
    elif f_name == "hemm":
        buffer += """
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of A is not referenced" ) {
        complex_t const _A[] = {{1, real_t(NAN)}, real_t(1), real_t(1), {1, real_t(NAN)}};
        REQUIRE_NOTHROW( hemm( layout, side, uplo, 2, 2, alpha, _A, 2, B, 2, beta, (complex_t*) C, 2 ) );
        CHECK( (C[0] == C[0] && C[1] == C[1] && C[2] == C[2] && C[3] == C[3]) ); // i.e., they are not NaN
        std::fill_n(C, """+sizeArray+""", 1);
    }}""",
        countCases += 1

    buffer += """
}""",   
    if countCases > 0 or countCasesNoChange > 0:
        print( "\n" + "".join(buffer) )
    else:
        print( "\n/*\n" + "".join(buffer) + " */" )
