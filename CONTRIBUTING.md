# Contributing guidelines

Thank you for investing your precious time to contribute to \<T\>LAPACK! Please read the following guidelines before proposing a pull request.

## Code style

### Automatic code formatting

\<T\>LAPACK uses [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) (version 10) to enforce a consistent code style. The configuration file is located at [`.clang-format`](.clang-format). The code style is based on the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) and the differences are marked in the configuration file. You should never modify the [`.clang-format`](.clang-format) file in \<T\>LAPACK, unless this is the reason behind your pull request.

To format code you are adding or modifying, we suggest using the [git-clang-format](https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/git-clang-format) script that is distributed together with ClangFormat. Use `git-clang-format` to format the code that was modified or added before a commit. You can use `git-clang-format --help` to access all options. Mind that one of the tests in the Continuous Integration checks that the code is properly formatted.

### Naming conventions

#### Template parameters and type aliases

Template parameters and type aliases should be named using the conventions:

1. Use `T` possibly followed by a short identifier (preferrably one uppercase character) to represent a scalar type like `float`, `double`, `int` and `std::complex<float>`. For instance, `TA` and `TB` could represent the types of variables `a` and `b`, or the types of the entries of matrices `A` and `B`.
2. Use `matrixA_t` and `vectorA_t` to represent the types of matrices and vectors, respectively. Use a short identifier (preferrably one uppercase character) like `A` and `B` between the words `matrix` (por `vector`) and `_t`.
3. Use `foo_bar_t` to represent the type of the variable `fooBar`. This convention should be used whenever the first and second rules are not clear enough. For instance, `alpha_t` and `beta_t` are used in [gemm](include/tlapack/blas/gemm.hpp) to represent the types of scalars `alpha` and `beta` in the BLAS function `gemm`, while `TA` and `TB` represent the types of the entries of matrices `A` and `B`.
4. Use `foo_bar_f` to represent the type of the functor `fooBar`. For instance, `abs_f` is the type of the functor `abs` in [lassq.hpp](include/tlapack/lapack/lassq.hpp).

Special cases:

- Use `real_t` to represent the default real type to be used inside a given scope.
- Use `scalar_t` or `T` to represent the default scalar type to be used inside a given scope. The identifier `T` is preferrable.
- Use `idx_t` to represent the default index type to be used inside a given scope.

#### Functions

Function arguments should be named using either:

1. a single capital letter if it is a matrix, or
2. lower camel case style.

For instance, `fooBar` is a good name for a function argument, while `foo_bar` is not. `A` and `matrixA` are good names for a matrix, while `a` is not.

#### Other naming conventions

1. Use upper camel case (Pascal case) style to name:
   - concepts, e.g., `tlapack::concepts::LegacyArray`.
   - enumeration classes, e.g., `tlapack::GetrfVariant`.
   - classes, e.g., `tlapack::ErrorCheck`.
2. Namespace variables are named using all-caps snake case, e.g., `tlapack::LOWER_HESSENBERG`.
3. Functions are named using snake case, e.g., `tlapack::multishift_qr`.
4. Preprocessor constants are named using all-caps snake case starting with `TLAPACK_`, e.g., `TLAPACK_NDEBUG`.
5. Preprocessor macros are named using lower snake case starting with `tlapack_`, e.g., `tlapack_check`.

Special cases:

1. Traits are classes named using snake case and have the suffix `_trait` or `_traits`, e.g., `tlapack::traits::entry_type_trait` and `tlapack::traits::matrix_type_traits`.
2. Classes for optional arguments of functions are named using upper camel case (Pascal case) style and have the suffix `Opts`, e.g., `tlapack::BlockedCholeskyOpts`.

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

are equivalent, and deduce the type `int` for `n`. We suggest avoiding the usage of `auto` in the following cases:

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

   b. slicing matrices and vectors using `tlapack::slice`, `tlapack::rows`, `tlapack::col`, etc. See [concepts](include/tlapack/base/concepts.hpp) for more details.

   c. the functor `tlapack::Create< >(...)`. See [arrayTraits.hpp](include/tlapack/base/arrayTraits.hpp) for more details.

2. In the return type of functions like `tlapack::asum`, `tlapack::dot`, `tlapack::nrm2` and `tlapack::lange`. By defining the output as `auto`, we enable overloading of those functions using mixed precision. For instance, one may write a overloading of `tlapack::lange` for matrices `Eigen::MatrixXf` that returns `double`.

### Good practices when writing code inside the library

1. Declare real-valued constants using real types. For instance, use `const real_type<T> zero(0)` instead of `const T zero(0)`. This is because the type `T` may be a complex type, and the constant `zero` is real. Avoid constants like `const real_type<T> rzero(0)` and `const T czero(0)` in the same function, because it is confusing.

2. Avoid any identifiers starting with underscores even if not followed by Uppercase. This is aligned to [17.4.3.1.2 C++ global names](https://timsong-cpp.github.io/cppwp/n3337/global.names). You may use the following regular expression to find relevant identifiers that start with underscore in VS Code:

   ```diff
   __(.*)__                      # Identifiers limited by double underscore
   ([^\w|])_([A-Z])              # Start with underscore then uppercase letter
   ([^\w])_(\w)([, ;/\)\(.\[\]]) # Other identifiers that start with underscore
   ```

3. In internal calls, use compile-time flags instead of runtime flags. For instance, use `tlapack::left_side` instead of `tlapack::Side::Left` and `tlapack::NO_TRANS` instead of `tlapack::Op::NoTrans`. This practice usually leads to faster code.

4. Avoid writing code that depends explicitly on `std::complex<T>` by using `tlapack::real_type<T>`, `tlapack::complex_type<T>` and `tlapack::scalar_type<T>`. Any scalar type `T` supported by \<T\>LAPACK should implement those 3 classes.

> **_NOTE:_** `std::complex` is one way to define complex numbers. It has undesired behavior such as:
>
> - `std::abs(std::complex<T>)` may not propagate NaNs. See https://github.com/tlapack/tlapack/issues/134#issue-1364091844.
> - Operations with `std::complex<T>`, for `T=float,double,long double` are wrappers to operations in C. Other types have their implementation in C++. Because of that, the logic of complex multiplication, division and other operations may change from type to type. See https://github.com/advanpix/mpreal/issues/11.

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

10. Do not allocate workspaces inside the test. Instead, let each routine allocate the workspace it needs. This has at least two benefits: (1) tests are simpler; (2) routines will run at minimum workspace, which means we will be testing that the minimum workspace size is well defined.

### Tags for tests

\<T\>LAPACK uses the following TAGs for tests:

TO-DO

## Other topics

### Updating [tests/blaspp](tests/blaspp) and [tests/lapackpp](tests/lapackpp)

There are two situations in which you may need to update [tests/blaspp](tests/blaspp) and [tests/lapackpp](tests/lapackpp):

1. When you want to enable a test.
2. When you want to use a new version of those libraries for tests.

### Running Github Actions locally

You can run the Github Actions locally using [act](https://github.com/nektos/act). This is useful for debugging the Github Actions workflow.
