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

    constexpr double ConstexprAbs(double x) { return x < 0.0 ? -x : x; }
    constexpr double ConstexprCeil(double x)
    {
        constexpr double kIntegerThreshold =
            uint64_t{1} << (std::numeric_limits<double>::digits - 1);
        // Too big or NaN inputs get returned unchanged.
        if (!(ConstexprAbs(x) < kIntegerThreshold)) {
            return x;
        }
        const double x_trunc = static_cast<double>(static_cast<int64_t>(x));
        return x_trunc < x ? x_trunc + 1.0 : x_trunc;
    }
    typedef ml_dtypes::float8_internal::float8_e4m3fn float8e4m3fn;

    inline float8e4m3fn ceil(float8e4m3fn x) noexcept
    {
        return float8e4m3fn(ConstexprCeil(double(x)));
    }
    inline float8e4m3fn floor(float8e4m3fn x) noexcept
    {
        return -ceil(float8e4m3fn(-1 * double(x)));
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

    typedef ml_dtypes::float8_internal::float8_e5m2fnuz float8e5m2fnuz;

    inline float8e5m2fnuz ceil(float8e5m2fnuz x) noexcept
    {
        return float8e5m2fnuz(ConstexprCeil(double(x)));
    }
    inline float8e5m2fnuz floor(float8e5m2fnuz x) noexcept
    {
        return -ceil(float8e5m2fnuz(-1 * double(x)));
    }
    inline float8e5m2fnuz log2(float8e5m2fnuz x) noexcept
    {
        return float8e5m2fnuz(log(double(x)));
    }
    inline float8e5m2fnuz max(float8e5m2fnuz x, float8e5m2fnuz y) noexcept
    {
        return x > y ? x : y;
    }
    inline float8e5m2fnuz min(float8e5m2fnuz x, float8e5m2fnuz y) noexcept
    {
        return x > y ? y : x;
    }
    inline float8e5m2fnuz sqrt(float8e5m2fnuz x) noexcept
    {
        return float8e5m2fnuz(std::sqrt(double(x)));
    }
    inline float8e5m2fnuz pow(int x, float8e5m2fnuz y)
    {
        return float8e5m2fnuz(std::pow(float(x), float(y)));
    }
    inline bool isinf(float8e5m2fnuz x)
    {
        return ml_dtypes::float8_internal::isinf(x);
    }
    typedef ml_dtypes::float8_internal::float8_e4m3fn float8e5m2fnuz;

    inline float8e5m2fnuz ceil(float8e5m2fnuz x) noexcept
    {
        return float8e5m2fnuz(ConstexprCeil(double(x)));
    }
    inline float8e5m2fnuz floor(float8e5m2fnuz x) noexcept
    {
        return -ceil(float8e5m2fnuz(-1 * double(x)));
    }
    inline float8e5m2fnuz log2(float8e5m2fnuz x) noexcept
    {
        return float8e5m2fnuz(log(double(x)));
    }
    inline float8e5m2fnuz max(float8e5m2fnuz x, float8e5m2fnuz y) noexcept
    {
        return x > y ? x : y;
    }
    inline float8e5m2fnuz min(float8e5m2fnuz x, float8e5m2fnuz y) noexcept
    {
        return x > y ? y : x;
    }
    inline float8e5m2fnuz sqrt(float8e5m2fnuz x) noexcept
    {
        return float8e5m2fnuz(std::sqrt(double(x)));
    }
    inline float8e5m2fnuz pow(int x, float8e5m2fnuz y)
    {
        return float8e5m2fnuz(std::pow(float(x), float(y)));
    }
    inline bool isinf(float8e5m2fnuz x)
    {
        return ml_dtypes::float8_internal::isinf(x);
    }
    
    typedef ml_dtypes::float8_internal::float8_e4m3b11 float8e4m3b11;

    inline float8e4m3b11 ceil(float8e4m3b11 x) noexcept
    {
        return float8e4m3b11(ConstexprCeil(double(x)));
    }
    inline float8e4m3b11 floor(float8e4m3b11 x) noexcept
    {
        return -ceil(float8e4m3b11(-1 * double(x)));
    }
    inline float8e4m3b11 log2(float8e4m3b11 x) noexcept
    {
        return float8e4m3b11(log(double(x)));
    }
    inline float8e4m3b11 max(float8e4m3b11 x, float8e4m3b11 y) noexcept
    {
        return x > y ? x : y;
    }
    inline float8e4m3b11 min(float8e4m3b11 x, float8e4m3b11 y) noexcept
    {
        return x > y ? y : x;
    }
    inline float8e4m3b11 sqrt(float8e4m3b11 x) noexcept
    {
        return float8e4m3b11(std::sqrt(double(x)));
    }
    inline float8e4m3b11 pow(int x, float8e4m3b11 y)
    {
        return float8e4m3b11(std::pow(float(x), float(y)));
    }
    inline bool isinf(float8e4m3b11 x)
    {
        return ml_dtypes::float8_internal::isinf(x);
    }
    typedef ml_dtypes::float8_internal::float8_e4m3fn float8e5m2fnuz;

    inline float8e5m2fnuz ceil(float8e5m2fnuz x) noexcept
    {
        return float8e5m2fnuz(ConstexprCeil(double(x)));
    }
    inline float8e5m2fnuz floor(float8e5m2fnuz x) noexcept
    {
        return -ceil(float8e5m2fnuz(-1 * double(x)));
    }
    inline float8e5m2fnuz log2(float8e5m2fnuz x) noexcept
    {
        return float8e5m2fnuz(log(double(x)));
    }
    inline float8e5m2fnuz max(float8e5m2fnuz x, float8e5m2fnuz y) noexcept
    {
        return x > y ? x : y;
    }
    inline float8e5m2fnuz min(float8e5m2fnuz x, float8e5m2fnuz y) noexcept
    {
        return x > y ? y : x;
    }
    inline float8e5m2fnuz sqrt(float8e5m2fnuz x) noexcept
    {
        return float8e5m2fnuz(std::sqrt(double(x)));
    }
    inline float8e5m2fnuz pow(int x, float8e5m2fnuz y)
    {
        return float8e5m2fnuz(std::pow(float(x), float(y)));
    }
    inline bool isinf(float8e5m2fnuz x)
    {
        return ml_dtypes::float8_internal::isinf(x);
    }
    
  }

/*
// e4m3fnuz
  namespace tlapack {
    namespace traits {
        template <>
        struct real_type_traits<ml_dtypes::float8_internal::float8_e4m3fnuz, int> {
            using type = ml_dtypes::float8_internal::float8_e4m3fnuz;
            constexpr static bool is_real = true;
        };
        template <>
        struct complex_type_traits<ml_dtypes::float8_internal::float8_e4m3fnuz, int> {
            using type = std::complex<ml_dtypes::float8_internal::float8_e4m3fnuz>;
            constexpr static bool is_complex = false;
        };
    }  // namespace traits

    constexpr double ConstexprAbs(double x) { return x < 0.0 ? -x : x; }
    constexpr double ConstexprCeil(double x)
    {
        constexpr double kIntegerThreshold =
            uint64_t{1} << (std::numeric_limits<double>::digits - 1);
        // Too big or NaN inputs get returned unchanged.
        if (!(ConstexprAbs(x) < kIntegerThreshold)) {
            return x;
        }
        const double x_trunc = static_cast<double>(static_cast<int64_t>(x));
        return x_trunc < x ? x_trunc + 1.0 : x_trunc;
    }
    typedef ml_dtypes::float8_internal::float8_e4m3fnuz float8e4m3fnuz;

    inline float8e4m3fnuz ceil(float8e4m3fnuz x) noexcept
    {
        return float8e4m3fnuz(ConstexprCeil(double(x)));
    }
    inline float8e4m3fnuz floor(float8e4m3fnuz x) noexcept
    {
        return -ceil(float8e4m3fnuz(-1 * double(x)));
    }
    inline float8e4m3fnuz log2(float8e4m3fnuz x) noexcept
    {
        return float8e4m3fnuz(log(double(x)));
    }
    inline float8e4m3fnuz max(float8e4m3fnuz x, float8e4m3fnuz y) noexcept
    {
        return x > y ? x : y;
    }
    inline float8e4m3fnuz min(float8e4m3fnuz x, float8e4m3fnuz y) noexcept
    {
        return x > y ? y : x;
    }
    inline float8e4m3fnuz sqrt(float8e4m3fnuz x) noexcept
    {
        return float8e4m3fnuz(std::sqrt(double(x)));
    }
    inline float8e4m3fnuz pow(int x, float8e4m3fnuz y)
    {
        return float8e4m3fnuz(std::pow(float(x), float(y)));
    }
    inline bool isinf(float8e4m3fnuz x)
    {
        return ml_dtypes::float8_internal::isinf(x);
    }
  }*/

// e5m2
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

  // namespace tlapack

inline std::istream& operator>>(std::istream& is, ml_dtypes::float8_e4m3fn& x)
{
    float f;
    is >> f;
    x = ml_dtypes::float8_e4m3fn(x);
    return is;
}
/*
inline std::istream& operator>>(std::istream& is, ml_dtypes::float8_e4m3fnuz& x)
{
    float f;
    is >> f;
    x = ml_dtypes::float8_e4m3fnuz(x);
    return is;
}*/

inline std::istream& operator>>(std::istream& is, ml_dtypes::float8_e5m2& x)
{
    float f;
    is >> f;
    x = ml_dtypes::float8_e5m2(x);
    return is;
}


inline std::istream& operator>>(std::istream& is, ml_dtypes::float8_e5m2fnuz& x)
{
    float f;
    is >> f;
    x = ml_dtypes::float8_e5m2fnuz(x);
    return is;
}

  // namespace std
