#pragma once
#include <type_traits>
#include <numbers>

constexpr float operator"" _deg(long double v) {
    return static_cast<float>(v * (std::numbers::pi / 180.0L));
}

constexpr float operator"" _deg(unsigned long long v) {
    return static_cast<float>(static_cast<long double>(v) * (std::numbers::pi / 180.0L));
}