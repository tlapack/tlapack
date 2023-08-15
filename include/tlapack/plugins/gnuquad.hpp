// using std::abs;
// using std::ceil;
// using std::floor;
// using std::isinf;
// using std::isnan;
// using std::log2;
// using std::max;
// using std::min;
// using std::pow;  // We only use pow(int, T), see below in the concept Real.
// using std::sqrt;

#include <quadmath.h>

#include <limits>

namespace std {

inline constexpr __float128 abs(__float128 x);  // See include/bits/std_abs.h
inline __float128 ceil(__float128 x) noexcept { return ceilq(x); }
inline __float128 floor(__float128 x) noexcept { return floorq(x); }
inline bool isinf(__float128 x) noexcept { return isinfq(x); }
inline bool isnan(__float128 x) noexcept { return isnanq(x); }
inline __float128 log2(__float128 x) noexcept { return log2q(x); }
inline __float128 max(__float128 x, __float128 y) noexcept
{
    return fmaxq(x, y);
}
inline __float128 min(__float128 x, __float128 y) noexcept
{
    return fminq(x, y);
}
inline __float128 pow(int x, __float128 y) noexcept { return powq(x, y); }
inline __float128 sqrt(__float128 x) noexcept { return sqrtq(x); }

template <>
struct numeric_limits<__float128> {
    static constexpr bool is_specialized = true;

    static constexpr __float128 min() noexcept { return FLT128_MIN; }
    static constexpr __float128 max() noexcept { return FLT128_MAX; }
    static constexpr __float128 lowest() noexcept { return -FLT128_MAX; }

    static constexpr int digits = FLT128_MANT_DIG;
    static constexpr int digits10 = FLT128_DIG;
    static constexpr int max_digits10 = (2 + FLT128_MANT_DIG * 643L / 2136);

    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;

    static constexpr int radix = 2;
    static constexpr __float128 epsilon() noexcept { return FLT128_EPSILON; }
    static constexpr __float128 round_error() noexcept { return 0.5F; }

    static constexpr int min_exponent = FLT128_MIN_EXP;
    static constexpr int min_exponent10 = FLT128_MIN_10_EXP;
    static constexpr int max_exponent = FLT128_MAX_EXP;
    static constexpr int max_exponent10 = FLT128_MAX_10_EXP;

    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    // static constexpr bool has_signaling_NaN = ;
    static constexpr float_denorm_style has_denorm = denorm_present;
    static constexpr bool has_denorm_loss = false;

    static constexpr __float128 infinity() noexcept
    {
        return __builtin_huge_valq();
    }
    static __float128 quiet_NaN() noexcept { return nanq(""); }
    // static constexpr __float128 signaling_NaN() noexcept {}
    static constexpr __float128 denorm_min() noexcept
    {
        return FLT128_DENORM_MIN;
    }

    // static constexpr bool is_iec559 = ;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;

    // static constexpr bool traps = ;
    // static constexpr bool tinyness_before = ;
    static constexpr float_round_style round_style = round_to_nearest;
};

}  // namespace std