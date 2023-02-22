# Contributing guidelines

Thank you for investing your precious time to contribute to \<T\>LAPACK! Please read the following guidelines before proposing a pull request.

## Code style

### Automatic code formatting

\<T\>LAPACK uses [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) (version 10) to enforce a consistent code style. The configuration file is located at [`.clang-format`](.clang-format). The code style is based on the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) and the differences are marked in the configuration file. You should never modify the [`.clang-format`](.clang-format) file in \<T\>LAPACK, unless this is the reason behind your pull request.

To format code you are adding or modifying, we suggest using the [git-clang-format](https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/git-clang-format) script that is distributed together with ClangFormat. Use `git-clang-format` to format the code that was modified or added before a commit. You can use `git-clang-format --help` to access all options. Mind that one of the tests in the Continuous Integration checks that the code is properly formatted.

### Usage of the `auto` keyword

In C++ all types are evaluated at compile time. The `auto` keyword is used to enable the compiler to deduce the type of a variable automatically. For instance, the instructions

```cpp
int m = 1;
auto n = m + m;
```

and 

```cpp
int m = 1;
int n = m + m;
```

are equivalent, and deduce the type `int` for `n`. The former is very useful and allows us to write generic functions and classes templated by other abstract classes. We suggest avoiding the usage of `auto` in the following cases:

1. When the type is known, like in [nrm2](include/tlapack/blas/nrm2.hpp):

    ```cpp
    const real_t tsml = blue_min<real_t>();
    ```

    In this case, the reader does not need to go to the definition of `blue_min` to know the type of `tsml`.

2. When the type can be deduced using `tlapack::real_type<>`, `tlapack::complex_type<>`, `tlapack::scalar_type<>`, `tlapack::type_t<>`, `tlapack::size_type<>` or any combination of them. For instance, in [gemm](include/tlapack/blas/gemm.hpp):

    ```cpp
    using TB = type_t<matrixB_t>;
    using scalar_t = scalar_type<alpha_t,TB>;

    const scalar_t alphaTimesblj = alpha*B(l,j);
    ```

    This design makes it clear the type we obtain after operating with the types. Moreover, it avoids a problem with lazy evaluation. Indeed, consider the following (bad) example:

    ```c++
    using T = mpf_class;
    T w = 1.0;
    T x = 2.0;
    auto y = w + x;
    T z = y * y;
    ```

    where `mpf_class` is the GNU multiprecision floating-point type. The type of `y` is not `mpf_class`, because `mpf_class` resort to lazy evaluation before evaluating mathematical expressions. Thus, `w + x` will be actually computed twice in the last instruction, which is not ideal. That is why we suggest to avoid the usage of `auto` in those situations.

We recommend the usage of `auto` in the following cases:

1. To get the output of

    a. `tlapack::legacy_matrix` and `tlapack::legacy_vector` when writing wrappers to optimized BLAS and LAPACK.

    b. slicing matrices and vectors using `tlapack::slice`, `tlapack::rows`, `tlapack::col`, etc. See [abstractArray](include/tlapack/plugins/abstractArray.hpp) for more details.
    
    c. the functor `tlapack::Create< >(...)`. See [arrayTraits.hpp](include/tlapack/base/arrayTraits.hpp) for more details.

2. In the return type of functions like `tlapack::asum`, `tlapack::dot`, `tlapack::nrm2` and `tlapack::lange`. By defining the output as `auto`, we enable overloading of those functions using mixed precision. For instance, one may write a overloading of `tlapack::lange` for matrices `Eigen::MatrixXf` that returns `double`.

## Writing tests

\<T\>LAPACK uses a couple of different test suites for unit tests, all located in the [test](test) directory. See [github.com/tlapack/tlapack/wiki/Test-Suite](https://github.com/tlapack/tlapack/wiki/Test-Suite) for more details on what each test set cover. Here we focus on how to write tests in [test/src](test/src) for \<T\>LAPACK.

We use [Catch2](https://github.com/catchorg/Catch2) as the testing framework. This means that you need to use a few macros like `TEST_CASE`, `REQUIRE`, `CHECK`, etc. It also means that you may use other macros provided by Catch2. For instance, you may use `REQUIRE_THROWS_AS` to check that a routine throws an exception of a certain type. See [github.com/catchorg/Catch2/docs](https://github.com/catchorg/Catch2/tree/devel/docs).

### Preparing the environment

Consider following the steps below before writing a test for a new routine:

1. Take a look at the current tests in [test/src](test/src) to see if there is already a test for the routine you want to add. If there is, you can add your test to the existing file. If there is not, you can create a new test file.

2. If you need to create a new test file, you can copy one of the existing tests. The naming convention is `test_<routine_name>.cpp`. For instance, the tests for `tlapack::getrf` are located in [test/src/test_getrf.cpp](test/src/test_getrf.cpp).

3. Add an entry `add_executable( test_<routine_name> test_<routine_name>.cpp )` to [test/src/CMakeLists.txt](test/src/CMakeLists.txt). This will make sure that the test is compiled and run when CTest is called.

4. Configure your CMake build to build the target `test_<routine_name>` so that you only need to compile and run the test you are writing.

### Directions for writing tests

1. Only include the headers you need. This also means:
    - You should never include `tlapack.hpp` in a test. Instead, include the headers for the routines you are testing. Compilation times may increase significantly if you include `tlapack.hpp` in a test.
    - If needed, include `testutils.hpp` before including other headers from \<T\>LAPACK. This will make sure that the macros are defined before they are used. Also, it makes sure that the plugins for matrix types are loaded before other headers are included.

2. You should use `using namespace tlapack;` in a test. This will make the code more readable.

3. If using the macro `TEMPLATE_TEST_CASE()` from Catch2, please set the last argument to either `TLAPACK_TYPES_TO_TEST`, `TLAPACK_REAL_TYPES_TO_TEST` or `TLAPACK_COMPLEX_TYPES_TO_TEST`. This will make sure that the test is run for all the types.

4. Try using tags following the directions in [Tags for tests](#tags-for-tests).

5. For creating objects inside the tests:

    - If you need to create a matrix, use `tlapack::Create<TestType>(...)`. See [arrayTraits.hpp](include/tlapack/base/arrayTraits.hpp) for more details.

    - If you need to create a vector there are two options. The first option is to use `tlapack::Create< vector_type<TestType> >(...)`. See [arrayTraits.hpp](include/tlapack/base/arrayTraits.hpp) for more details. The secon optionis to use `std::vector< type_t<TestType> >` or `std::vector< real_type<type_t<TestType>> >`.

6. Use the macro `GENERATE()` to create a range of values. For instance, you may use `GENERATE(1,2,3)` to create a range of values `{1,2,3}`. This way you can avoid writing a loop to test a routine for different values of a parameter.

7. Whenever possible, create a `DYNAMIC_SECTION(...)` after all commands `GENERATE()`. Example:

    ```c++
    const idx_t m = GENERATE(1,5,9);
    const idx_t n = GENERATE(1,5,9);
    const Uplo = GENERATE(Uplo::Lower, Uplo::Upper);

    DYNAMIC_SECTION("m = " << m << " n = " << n << " Uplo = " << Uplo) {
        // test code
    }
    ```

8. Use the macros `INFO()` and `UNSCOPED_INFO()` to print other information about the test.

9. Use the macros `REQUIRE()`, `CHECK()`, `REQUIRE_THROWS_AS()` and `CHECK_THROWS_AS()` to check the results of the test. `CHECK()` is preferred over `REQUIRE()` because it allows the test to continue even if one of the assertions is false. Use `REQUIRE()` only when it does not make sense to continue the test if the assertion is false.

### Tags for tests

\<T\>LAPACK uses the following TAGs for tests:

TO-DO

## Other topics

### Updating [tests/blaspp](tests/blaspp) and [tests/lapackpp](tests/lapackpp)

There are two situations in which you may need to update [tests/blaspp](tests/blaspp) and [tests/lapackpp](tests/lapackpp):
1. When you want to enable a test.
2. When you want to use a new version of those libraries for tests.

### Swap

Although they have the same name, `std::swap` and `tlapack::swap` do different things.

- `std::swap` takes 2 objects of the same type T and swaps either (1) their values, if T is a basic type; or (2) all their attributes, if T is a class.

- `tlapack::swap` takes 2 vectors of types `vectorX_t` and `vectorY_t` and swaps their entries.

> which swap should we use tlapack/blas/swap.hpp or std::swap ?
> getrf_recursive seems to include tlapack/blas/swap.hpp and use std::swap

Use `tlapack::swap` whenever you want to swap the entries between two arrays.
Currently, `std::swap` is only used in \<T\>LAPACK to avoid a code like 
```c++
// Swaps a and b. Equivalent to std::swap(a,b)
T aux = a;
a = b;
b = aux;
```

> Do we use std::swap to swap two scalars and tlapack/blas/swap.hpp to swap vectors?

Yes, that is how we currently use those functions.


TO-DO
