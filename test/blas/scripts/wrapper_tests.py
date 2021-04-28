#!/usr/bin/env python3
## @brief Create and populate 'test_corner_cases.cpp'
#
#  @author Weslley S. Pereira, University of Colorado Denver
#  @date   March 30, 2021

# Test if s can be converted into an integer
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

# BLAS routines' definitions
blas_routines = open("blas_routines.csv").read().splitlines()
for i in range(len(blas_routines)):
    blas_routines[i] = list(filter(None, [x.strip() for x in blas_routines[i].split(',')]))

# Functions that only accepts real types
realOnly_funcs = ["rot", "rotm", "rotmg", "syr", "symv"]

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

# Print header of the test file:
print("""\
#include <catch2/catch.hpp>
#include <tblas.hpp>
#include "test_types.hpp"

using namespace blas;""")

# Loop in the functions
for i, line in enumerate(blas_routines):
    f_name = line[0]
    blas_lv = int(line[1])
    ref_types = line[2::2]
    ref_args = line[3::2]

    buffer = """\
TEMPLATE_TEST_CASE( \"""" + f_name + """ satisfies all corner cases", "[""" + \
        f_name + """][BLASlv""" + str(blas_lv) + """]", """ + \
        ("TEST_TYPES" if f_name not in realOnly_funcs else "TEST_REAL_TYPES") + \
        """ ) {
    
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
        elif isArray:
            buffer += " = {1, 1, 1, 1};",
        else:
            buffer += " = 1;",
            
    buffer += """

    // Corner cases:""",
    countCases = 0

    # Tests that throws an Error Exception:
    throwExceptionBuffer = ()
    for k in throwException_corner_tests[f_name]:
        if RepresentsInt(k):
            configStr = throwException_corner_rules[int(k)][0]
        else:
            configStr = k
        try:
            attribs = [x.strip() for x in configStr.split(';')]
            args = ref_args.copy()
            for varAttrib in attribs:
                param, value = [x.strip() for x in varAttrib.split('=', 1)]
                for i, x in enumerate(args):
                    if x == param:
                        args[i] = value
                        break
        except: # Invalid test
            continue
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
        """ + ref_types[j].replace("*", "") + "ref_"+arg + "[] = {"+arg+"[0], "+arg+"[1], "+arg+"[2], "+arg+"[3]};",
            noChangeBuffer += "std::equal("+arg+", "+arg+"+4, ref_"+arg+")",
            swapBuffer += """
        std::swap( """ + arg + ", ref_" + arg + " );",
    noChangeStr = "("+" && ".join(noChangeBuffer)+")"
    refVarStr = "".join(refVarBuffer)
    swapStr = "".join(swapBuffer)

    for k in returnImmediately_corner_tests[f_name]:
        if RepresentsInt(k):
            configStr = returnImmediately_corner_rules[int(k)][0]
        else:
            configStr = k
        try:
            attribs = [x.strip() for x in configStr.split(';')]
            args = ref_args.copy()
            for varAttrib in attribs:
                param, value = [x.strip() for x in varAttrib.split('=', 1)]
                for i, x in enumerate(args):
                    if x == param:
                        args[i] = value
                        break
        except: # Invalid test
            continue

        bufferRequireNoChanges += """
    SECTION( \"""" + configStr + """\" ) {""" + refVarStr + """
        REQUIRE_NOTHROW( """ + f_name + "( " + ", ".join(args) + """ ) );
        CHECK( """ + noChangeStr + """ );""" + swapStr + """
    }""",
        countCasesNoChange += 1

    if countCasesNoChange > 0:
        buffer += bufferRequireNoChanges

    # Specific tests Lv1:
    if f_name == "dot" or f_name == "dotu":
        buffer += """
    SECTION ( "n <= 0" ) {
        CHECK( """+f_name+"""(-1, x, incx, y, incy ) == real_type<TestType>(0) );
        CHECK( """+f_name+"""( 0, x, incx, y, incy ) == real_type<TestType>(0) );
    }""",
        countCases += 2
    elif f_name == "asum" or f_name == "nrm2":
        buffer += """
    SECTION ( "n <= 0" ) {
        CHECK( """+f_name+"""(-1, x, incx ) == real_type<TestType>(0) );
        CHECK( """+f_name+"""( 0, x, incx ) == real_type<TestType>(0) );
    }""",
        countCases += 2
    elif f_name == "iamax":
        buffer += """
    SECTION ( "n <= 0" ) {
        CHECK( iamax(-1, x, incx ) == INVALID_INDEX );
        CHECK( iamax( 0, x, incx ) == INVALID_INDEX );
    }""",
        countCases += 2

    # Specific tests Lv2:
    elif f_name == "gemv":
        buffer += """
    SECTION( "y does not need to be set if beta = 0" ) {
        y[0] = y[1] = NAN;
        REQUIRE_NOTHROW( gemv( layout, trans, 2, 2, alpha, A, 2, x, incx, 0, y, incy ) );
        CHECK( (y[0] == y[0] && y[1] == y[1]) ); // i.e., they are not NaN
        y[0] = y[1] = 1;
    }""",
        countCases += 1
    elif f_name == "hemv":
        buffer += """
    SECTION( "y does not need to be set if beta = 0" ) {
        y[0] = y[1] = NAN;
        REQUIRE_NOTHROW( hemv( layout, uplo, 2, alpha, A, 2, x, incx, 0, y, incy ) );
        CHECK( (y[0] == y[0] && y[1] == y[1]) ); // i.e., they are not NaN
        y[0] = y[1] = 1;
    }
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of A is not referenced" ) {
        complex_t const _A[] = {{1, NAN}, 1, 1, {1, NAN}};
        REQUIRE_NOTHROW( hemv( layout, uplo, 2, alpha, _A, 2, (complex_t const *)x, incx, beta, (complex_t *)y, incy ) );
        CHECK( (y[0] == y[0] && y[1] == y[1]) ); // i.e., they are not NaN
        y[0] = y[1] = 1;
    }}""",
        countCases += 2
    elif f_name == "symv":
        buffer += """
    SECTION( "y does not need to be set if beta = 0" ) {
        y[0] = y[1] = NAN;
        REQUIRE_NOTHROW( symv( layout, uplo, 2, alpha, A, 2, x, incx, 0, y, incy ) );
        CHECK( (y[0] == y[0] && y[1] == y[1]) ); // i.e., they are not NaN
        y[0] = y[1] = 1;
    }""",
        countCases += 1
    elif f_name == "her":
        buffer += """
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of A is zero" ) {
        complex_t _A[] = {{1, NAN}, 1, 1, {1, NAN}};
        REQUIRE_NOTHROW( her( layout, uplo, 2, alpha, (complex_t const *)x, incx, _A, 2 ) );
        CHECK( (_A[0] == _A[0] && _A[1] == _A[1] && _A[2] == _A[2] && _A[3] == _A[3]) ); // i.e., they are not NaN
        CHECK( (std::imag(_A[0]) == TestType(0) && std::imag(_A[3]) == TestType(0)) );
    }}""",
        countCases += 1
    elif f_name == "her2":
        buffer += """
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of A is zero" ) {
        complex_t _A[] = {{1, NAN}, 1, 1, {1, NAN}};
        REQUIRE_NOTHROW( her2( layout, uplo, 2, alpha, (complex_t const *)x, incx, y, incy, _A, 2 ) );
        CHECK( (_A[0] == _A[0] && _A[1] == _A[1] && _A[2] == _A[2] && _A[3] == _A[3]) ); // i.e., they are not NaN
        CHECK( (std::imag(_A[0]) == TestType(0) && std::imag(_A[3]) == TestType(0)) );
    }}""",
        countCases += 1
    elif f_name == "trmv" or f_name == "trsv":
        buffer += """
    SECTION( "Diagonal of A is not referenced if diag = 'U'" ) {
        TestType const _A[] = {NAN, 1, 1, NAN};
        REQUIRE_NOTHROW( """+f_name+"""( layout, uplo, trans, Diag('U'), 2, _A, 2, x, incx ) );
        CHECK( (x[0] == x[0] && x[1] == x[1]) ); // i.e., they are not NaN
        x[0] = x[1] = 1;
    }""",
        countCases += 1

    # Specific tests Lv3:
    elif f_name == "gemm":
        buffer += """
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = TestType(2)*C[0];
        REQUIRE_NOTHROW( gemm( layout, transA, transB, m, n, 0, alpha, A, lda, B, ldb, real_type<TestType>(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }""",
        countCases += 1
    elif f_name == "syrk":
        buffer += """
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = TestType(2)*C[0];
        REQUIRE_NOTHROW( """+f_name+"""( layout, uplo, trans, n, 0, alpha, A, lda, real_type<TestType>(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }""",
        countCases += 1
    elif f_name == "syr2k":
        buffer += """
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = TestType(2)*C[0];
        REQUIRE_NOTHROW( """+f_name+"""( layout, uplo, trans, n, 0, alpha, A, lda, B, ldb, real_type<TestType>(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }""",
    elif f_name == "herk":
        buffer += """
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = TestType(2)*C[0];
        REQUIRE_NOTHROW( """+f_name+"""( layout, uplo, trans, n, 0, alpha, A, lda, real_type<TestType>(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of C is zero" ) {
        complex_t _C[] = {{1, NAN}, 1, 1, {1, NAN}};
        REQUIRE_NOTHROW( herk( layout, uplo, trans, 2, 2, alpha, A, 2, beta, _C, 2 ) );
        CHECK( std::equal(_C, _C+4, _C) ); // i.e., they C does not have NaN
    }}""",
        countCases += 1
    elif f_name == "her2k":
        buffer += """
    SECTION( "C := beta C if M, N > 0 and K = 0" ) {
        TestType const C11 = TestType(2)*C[0];
        REQUIRE_NOTHROW( """+f_name+"""( layout, uplo, trans, n, 0, alpha, A, lda, B, ldb, real_type<TestType>(2), C, ldc ) );
        CHECK( C[0] == C11 );
        C[0] = C[1] = C[2] = C[3] = 1;
    }
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of C is zero" ) {
        complex_t _C[] = {{1, NAN}, 1, 1, {1, NAN}};
        REQUIRE_NOTHROW( her2k( layout, uplo, trans, 2, 2, alpha, A, 2, B, 2, beta, _C, 2 ) );
        CHECK( std::equal(_C, _C+4, _C) ); // i.e., they C does not have NaN
    }}""",
        countCases += 1
    elif f_name == "hemm":
        buffer += """
    if (is_complex<TestType>::value) {
    using complex_t = complex_type<TestType>;
    SECTION( "Imaginary part of the diagonal of A is not referenced" ) {
        complex_t const _A[] = {{1, NAN}, 1, 1, {1, NAN}};
        REQUIRE_NOTHROW( hemm( layout, side, uplo, 2, 2, alpha, _A, 2, B, 2, beta, (complex_t*) C, 2 ) );
        CHECK( std::equal(C, C+4, C) ); // i.e., they C does not have NaN
        std::fill_n(C, 4, 1);
    }}""",
        countCases += 1

    buffer += """
}""",   
    if countCases > 0 or countCasesNoChange > 0:
        print( "\n" + "".join(buffer) )
    else:
        print( "\n/*\n" + "".join(buffer) + " */" )
