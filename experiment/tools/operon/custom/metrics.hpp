// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef METRICS_HPP
#define METRICS_HPP

#include "types.hpp"
#include "contracts.hpp"

#include <type_traits>
#include <vstat.hpp>

namespace Operon {

inline const auto SquaredError = [](auto a, auto b) { 
    auto e = a - b; 
    return e * e; };

template<typename InputIt1, typename InputIt2>
inline auto CoefficientOfDetermination(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    using V2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<V1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<V2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<V1, V2>, "The types must be the same");
    constexpr double eps{1e-12};
    auto ssr = univariate::accumulate<V1>(begin1, end1, begin2, SquaredError).sum;
    auto end2 = begin2;
    std::advance(end2, std::distance(begin1, end1));
    auto meanY = univariate::accumulate<V2>(begin2, end2).mean;
    auto sst = univariate::accumulate<V2>(begin2, end2, [&](auto v) { return SquaredError(v, meanY);} ).sum;
    if (sst < eps) {
        return std::numeric_limits<double>::min();
    }
    return 1.0 - ssr / sst;
}

template<typename InputIt1, typename InputIt2>
inline auto MeanSquaredError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    using V2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<V1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<V2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<V1, V2>, "The types must be the same");
    return univariate::accumulate<V1>(begin1, end1, begin2, SquaredError).mean;
}

template<typename InputIt1, typename InputIt2>
inline auto RootMeanSquaredError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    return std::sqrt(MeanSquaredError(begin1, end1, begin2));
}

template<typename InputIt1, typename InputIt2>
inline auto NormalizedMeanSquaredError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    using V2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<V1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<V2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<V1, V2>, "The types must be the same");

    constexpr double eps{1e-12};
    auto varY = univariate::accumulate<V1>(begin2, begin2 + std::distance(begin1, end1)).variance;
    if (std::abs(varY) < eps) {
        return varY;
    }
    return MeanSquaredError(begin1, end1, begin2) / varY;
}

template<typename InputIt1, typename InputIt2>
inline auto MeanAbsoluteError(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    using V2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<V1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<V2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<V1, V2>, "The types must be the same");
    return univariate::accumulate<V1>(begin1, end1, begin2, [](auto a, auto b) { return std::abs(a-b); }).mean;
}

template<typename InputIt1, typename InputIt2>
inline auto SquaredCorrelation(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    using V2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<V1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<V2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<V1, V2>, "The types must be the same");
    auto r = bivariate::accumulate<V1>(begin1, end1, begin2).correlation;
    return r * r;
}

template<typename InputIt1, typename InputIt2>
inline auto L2Norm(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) noexcept -> double
{
    using V1 = typename std::iterator_traits<InputIt1>::value_type;
    using V2 = typename std::iterator_traits<InputIt2>::value_type;
    static_assert(std::is_arithmetic_v<V1>, "InputIt1: value_type must be arithmetic.");
    static_assert(std::is_arithmetic_v<V2>, "InputIt2: value_type must be arithmetic.");
    static_assert(std::is_same_v<V1, V2>, "The types must be the same");

    auto sum = univariate::accumulate<V1>(begin1, end1, begin2, SquaredError).sum;
    return sum / 2;
}

template<typename T>
inline auto CoefficientOfDetermination(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> double
{
    EXPECT(y.size() == x.size());
    return CoefficientOfDetermination(x.begin(), x.end(), y.begin());
}

template<typename T>
inline auto MeanSquaredError(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> double
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    return univariate::accumulate<T>(x.data(), y.data(), x.size(), SquaredError).mean;
}

template<typename T>
inline auto RootMeanSquaredError(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> double
{
    return std::sqrt(MeanSquaredError(x, y));
}

template<typename T>
inline auto NormalizedMeanSquaredError(Operon::Span<T const> x, Operon::Span<T const> y) noexcept -> double
{
    return NormalizedMeanSquaredError(x.begin(), x.end(), y.begin());
}

template<typename T>
inline auto MeanAbsoluteError(Operon::Span<T const> x, Operon::Span<T const> y) -> double
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    return univariate::accumulate<T>(x.data(), y.data(), x.size(), [](auto a, auto b) { return std::abs(a-b); }).mean;
}

template<typename T>
inline auto L2Norm(Operon::Span<T const> x, Operon::Span<T const> y) -> double
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> a(x.data(), x.size());
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> b(y.data(), y.size());
    return static_cast<double>((a - b).squaredNorm() / 2);
}

template<typename T>
inline auto SquaredCorrelation(Operon::Span<T const> x, Operon::Span<T const> y) -> double
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    EXPECT(x.size() == y.size());
    EXPECT(x.size() > 0);
    auto r = bivariate::accumulate<T>(x.data(), y.data(), x.size()).correlation;
    return r * r;
}

// functors to plug into an evaluator
struct MSE {
    template<typename T>
    inline auto operator()(Operon::Span<T const> x, Operon::Span<T const> y) const noexcept -> double
    {
        return MeanSquaredError(x, y);
    }

    template<typename InputIt1, typename InputIt2>
    inline auto operator()(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) const noexcept -> double
    {
        return MeanSquaredError(begin1, end1, begin2);
    }
};

struct NMSE {
    template<typename T>
    inline auto operator()(Operon::Span<T const> x, Operon::Span<T const> y) const noexcept -> double
    {
        return NormalizedMeanSquaredError(x, y);
    }

    template<typename InputIt1, typename InputIt2>
    inline auto operator()(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) const noexcept -> double
    {
        return NormalizedMeanSquaredError(begin1, end1, begin2);
    }
};

struct RMSE {
    template<typename T>
    inline auto operator()(Operon::Span<T const> x, Operon::Span<T const> y) const noexcept -> double
    {
        return RootMeanSquaredError(x, y);
    }

    template<typename InputIt1, typename InputIt2>
    inline auto operator()(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) const noexcept -> double
    {
        return RootMeanSquaredError(begin1, end1, begin2);
    }
};

struct MAE {
    template<typename T>
    inline auto operator()(Operon::Span<T const> x, Operon::Span<T const> y) const noexcept -> double
    {
        return MeanAbsoluteError(x, y);
    }

    template<typename InputIt1, typename InputIt2>
    inline auto operator()(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) const noexcept -> double
    {
        return MeanAbsoluteError(begin1, end1, begin2);
    }
};

struct C2 {
    template<typename T>
    inline auto operator()(Operon::Span<T const> x, Operon::Span<T const> y) const noexcept -> double
    {
        return -SquaredCorrelation(x, y);
    }

    template<typename InputIt1, typename InputIt2>
    inline auto operator()(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) const noexcept -> double
    {
        return -SquaredCorrelation(begin1, end1, begin2);
    }
};

struct L2 {
    template<typename T>
    inline auto operator()(Operon::Span<T const> x, Operon::Span<T const> y) const noexcept -> double
    {
        return L2Norm(x, y);
    }

    template<typename InputIt1, typename InputIt2>
    inline auto operator()(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) const noexcept -> double
    {
        return L2Norm(begin1, end1, begin2);
    }
};

struct R2 {
    template<typename T>
    inline auto operator()(Operon::Span<T const> x, Operon::Span<T const> y) const noexcept -> double
    {
        return CoefficientOfDetermination(x, y);
        // return -CoefficientOfDetermination(x, y);
    }

    template<typename InputIt1, typename InputIt2>
    inline auto operator()(InputIt1 begin1, InputIt1 end1, InputIt2 begin2) const noexcept -> double
    {
        return CoefficientOfDetermination(begin1, end1, begin2);
        // return -CoefficientOfDetermination(begin1, end1, begin2);
    }
};

} // namespace Operon
#endif