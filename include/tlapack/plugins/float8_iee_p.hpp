// #ifndef TLAPACK_FLOAT8_HH
// #define TLAPACK_FLOAT8_HH
#include <math.h>
#include "float8.h"
#include <complex>
#include <limits>
#include <ostream>


namespace std{
    
   namespace tlapack{
    constexpr double ConstexprAbs(double x) { return x < 0.0 ? -x : x; }
    constexpr double ConstexprCeil(double x) {
  constexpr double kIntegerThreshold =
      uint64_t{1} << (std::numeric_limits<double>::digits - 1);
  // Too big or NaN inputs get returned unchanged.
  if (!(ConstexprAbs(x) < kIntegerThreshold)) {
    return x;
  }
  const double x_trunc = static_cast<double>(static_cast<int64_t>(x));
  return x_trunc < x ? x_trunc + 1.0 : x_trunc;
}
    typedef ml_dtypes::float8_e4m3fn float8e4m3fn;
inline float8e4m3fn abs(float8e4m3fn x) noexcept { return ml_dtypes::float8_internal::abs(x);}
inline float8e4m3fn ceil(float8e4m3fn x) noexcept {
   return  float8e4m3fn( ConstexprCeil(double(x)));
}
inline float8e4m3fn floor(float8e4m3fn x) noexcept { return -ceil(float8e4m3fn(-1*double(x))); }
inline float8e4m3fn log2(float8e4m3fn x) noexcept { return float8e4m3fn(log(double(x)));}
inline float8e4m3fn max(float8e4m3fn x, float8e4m3fn y) noexcept
{
   return float8e4m3fn((double(x) > double(y))? x : y);
}
inline float8e4m3fn min(float8e4m3fn x, float8e4m3fn y) noexcept
{
   return float8e4m3fn((double(x) > double(y))? y : x);
}
inline float8e4m3fn sqrt(float8e4m3fn x) noexcept {
   return float8e4m3fn(std::sqrt(double(x)));
}
inline float8e4m3fn pow(int x, float8e4m3fn y) {
   return float8e4m3fn(std::pow(float(x),float(y)));
}
   }




inline std::istream& operator>>(std::istream& is, ml_dtypes::float8_e4m3fn& x){

    float f;
    is >> f;
    x = ml_dtypes::float8_e4m3fn(x);
    return is;

}



}
