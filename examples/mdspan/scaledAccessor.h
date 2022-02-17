
#include <experimental/mdspan>

// -----------------------------------------------------------------------------
/** Scaled accessor policy for mdspan.
 * @brief Allows for the lazy evaluation of the scale operation of arrays
 * 
 * @tparam ElementType  Type of the data
 * @tparam scalar_t     Type of the scalar multiplying the data
 */
template <class ElementType, class scalar_t>
struct scaled_accessor {
  
    using offset_policy = scaled_accessor;
    using element_type = ElementType;
    using pointer = ElementType*;
    using scalar_type = scalar_t;
    using scalar_ptr = scalar_t*;

    inline constexpr scaled_accessor() noexcept
    : scale_( 1 ) { };

    inline constexpr scaled_accessor( const scalar_t& scale ) noexcept
    : scale_( scale ) { };

    template< class OtherElementType, class otherScalar_t,
        enable_if_t< (
            is_convertible_v<
                typename scaled_accessor<OtherElementType, otherScalar_t>::pointer, 
                pointer >
            &&
            is_convertible_v< 
                typename scaled_accessor<OtherElementType, otherScalar_t>::scalar_ptr, 
                scalar_ptr >
        ), bool > = true
    >
    inline constexpr scaled_accessor( scaled_accessor<OtherElementType, otherScalar_t> ) noexcept {}

    inline constexpr pointer
    offset( pointer p, std::size_t i ) const noexcept {
        return p + i;
    }

    inline constexpr auto access(pointer p, std::size_t i) const noexcept {
        return scale_ * p[i];
    }

    inline constexpr scalar_type scale() const noexcept {
        return scale_;
    }

    private:
        scalar_t scale_;
};

/**
 * @brief scale an array by a scalar alpha
 * 
 * It is a lazy evaluation strategy. The actual scale occurs element-wisely at the evaluation on each 
 * 
 * @tparam scalar_t Type of the scalar multiplying the data
 * @tparam array_t  Type of the array
 * @param alpha     Scalar
 * @param A         Array
 * @return An array   
 */
template<
    class scalar_t, class T, std::size_t... Exts, class LP, class AP,
    enable_if_t< is_same_v< AP, default_accessor<T> >
    , bool > = true
>
constexpr auto scale(
    const scalar_t& alpha, 
    const mdspan<T, std::experimental::extents<Exts...>, LP, AP>& A )
{
    using newAP = scaled_accessor<T,scalar_t>;
    return mdspan<T, std::experimental::extents<Exts...>, LP, newAP>(
        A.data(), A.mapping(), newAP( alpha )
    );
}

template<
    class scalar_t, class T, std::size_t... Exts, class LP, class AP, class otherScalar_t,
    enable_if_t< is_same_v< AP, scaled_accessor<T,otherScalar_t> >
    , bool > = true
>
constexpr auto scale(
    const scalar_t& alpha, 
    const mdspan<T, std::experimental::extents<Exts...>, LP, AP>& A )
{
    const auto beta = alpha * A.accessor().scale();
    using newAP = scaled_accessor<T,decltype(beta)>;
    return mdspan<T, std::experimental::extents<Exts...>, LP, newAP>(
        A.data(), A.mapping(), newAP( beta )
    );
}

template<
    class scalar_t, class T, std::size_t... Exts, class LP, class AP, class otherScalar_t,
    enable_if_t< is_same_v< AP, scaled_accessor<T,otherScalar_t> >
    , bool > = true
>
constexpr auto scale(
    const scalar_t& alpha, 
    const mdspan<T, std::experimental::extents<Exts...>, LP, AP>& A )
{
    const auto beta = alpha * A.accessor().scale();
    using newAP = scaled_accessor<T,decltype(beta)>;
    return mdspan<T, std::experimental::extents<Exts...>, LP, newAP>(
        A.data(), A.mapping(), newAP( beta )
    );
}