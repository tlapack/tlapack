# Contributing guidelines

Thank you for investing your precious time to contribute to \<T\>LAPACK! Please read the following guidelines before proposing a pull request.

## Code style

### Automatic code formatting

\<T\>LAPACK uses [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) (version 10) to enforce a consistent code style. The configuration file is located at [`.clang-format`](.clang-format). The code style is based on the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) and the differences are marked in the configuration file. You should never modify the [`.clang-format`](.clang-format) file in \<T\>LAPACK, unless this is the reason behind your pull request.

To format code you are adding or modifying, we suggest using the [git-clang-format](https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/git-clang-format) script that is distributed together with ClangFormat. Use `git clang-format` to format the code that was modified or added before a commit. You can use `git-clang-format --help` to access all options. Mind that one of the tests in the Continuous Integration checks that the code is properly formatted.

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
