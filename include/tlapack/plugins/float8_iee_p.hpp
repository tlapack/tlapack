// #ifndef TLAPACK_FLOAT8_HH
// #define TLAPACK_FLOAT8_HH
#include <math.h>

#include <complex>
#include <limits>
#include <ostream>
#include <type_traits>

#include "../../../eigen/Eigen/Core"
#include "float8.h"
#include "tlapack/base/scalar_type_traits.hpp"
#include "tlapack/base/types.hpp"

using namespace std;
using namespace ml_dtypes::float8_internal;

namespace tlapack {
    namespace traits {
        template <>
        struct real_type_traits<ml_dtypes::float8_internal::float8_e4m3fn, int> {
            using type = ml_dtypes::float8_internal::float8_e4m3fn;
            constexpr static bool is_real = true;
        };
        template <>
        struct complex_type_traits<ml_dtypes::float8_internal::float8_e4m3fn, int> {
            using type = std::complex<ml_dtypes::float8_internal::float8_e4m3fn>;
            constexpr static bool is_complex = false;
        };
    }  // namespace traits

  }

namespace tlapack {
    namespace traits {
        template <>
        struct real_type_traits<ml_dtypes::float8_internal::float8_e5m2, int> {
            using type = ml_dtypes::float8_internal::float8_e5m2;
            constexpr static bool is_real = true;
        };
        template <>
        struct complex_type_traits<ml_dtypes::float8_internal::float8_e5m2, int> {
            using type = std::complex<ml_dtypes::float8_internal::float8_e5m2>;
            constexpr static bool is_complex = false;
        };
    }  // namespace traits

    
  }

  // namespace tlapack

inline std::istream& operator>>(std::istream& is, ml_dtypes::float8_e4m3fn& x)
{
    float f;
    is >> f;
    x = ml_dtypes::float8_e4m3fn(x);
    return is;
}

inline std::istream& operator>>(std::istream& is, ml_dtypes::float8_e5m2& x)
{
    float f;
    is >> f;
    x = ml_dtypes::float8_e5m2(x);
    return is;
}

  // namespace std

  namespace ml_dtypes{
    namespace float8_internal {
        typedef float8_e4m3fn float8e4m3fn;
        inline float8e4m3fn ceil(float8e4m3fn x) noexcept
    {
        return float8e4m3fn(ConstexprCeil(double(x)));
    }
    inline float8e4m3fn floor(float8e4m3fn x) noexcept
    {
        return float8e4m3fn(-ConstexprCeil(-1 * double(x)));
    }
    inline float8e4m3fn log2(float8e4m3fn x) noexcept
    {
        return float8e4m3fn(log(double(x)));
    }
    inline float8e4m3fn max(float8e4m3fn x, float8e4m3fn y) noexcept
    {
        return x > y ? x : y;
    }
    inline float8e4m3fn min(float8e4m3fn x, float8e4m3fn y) noexcept
    {
        return x > y ? y : x;
    }
    inline float8e4m3fn sqrt(float8e4m3fn x) noexcept
    {
        return float8e4m3fn(std::sqrt(double(x)));
    }
    inline float8e4m3fn pow(int x, float8e4m3fn y)
    {
        return float8e4m3fn(std::pow(float(x), float(y)));
    }
    inline bool isinf(float8e4m3fn x)
    {
        return ml_dtypes::float8_internal::isinf(x);
    }
    typedef ml_dtypes::float8_internal::float8_e5m2 float8e5m2;

    inline float8e5m2 ceil(float8e5m2 x) noexcept
    {
        return float8e5m2(ConstexprCeil(double(x)));
    }
    inline float8e5m2 floor(float8e5m2 x) noexcept
    {
        return -ceil(float8e5m2(-1 * double(x)));
    }
    inline float8e5m2 log2(float8e5m2 x) noexcept
    {
        return float8e5m2(log(double(x)));
    }
    inline float8e5m2 max(float8e5m2 x, float8e5m2 y) noexcept
    {
        return x > y ? x : y;
    }
    inline float8e5m2 min(float8e5m2 x, float8e5m2 y) noexcept
    {
        return x > y ? y : x;
    }
    inline float8e5m2 sqrt(float8e5m2 x) noexcept
    {
        return float8e5m2(std::sqrt(double(x)));
    }
    inline float8e5m2 pow(int x, float8e5m2 y)
    {
        return float8e5m2(std::pow(float(x), float(y)));
    }
    inline bool isinf(float8e5m2 x)
    {
        return ml_dtypes::float8_internal::isinf(x);
    }
    }
    }