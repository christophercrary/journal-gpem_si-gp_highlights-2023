// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef NODE_HPP
#define NODE_HPP

#include <array>
#include <bitset>
#include <cstdint>
#include <type_traits>

#include "types.hpp"
#include <fmt/format.h>

namespace Operon {
enum class NodeType : uint32_t {
    // terminal nodes - ordered by arity
    Add = 1u << 0,
    Mul = 1u << 1,
    Sub = 1u << 2,
    Div = 1u << 3,
    Aq  = 1u << 4,
    Pow = 1u << 5,
    Log = 1u << 6,
    Exp = 1u << 7,
    Sin = 1u << 8,
    Cos = 1u << 9,
    Tan = 1u << 10,
    Tanh = 1u << 11,
    Sqrt = 1u << 12,
    Cbrt = 1u << 13,

    Ln = 1u <<14,

    Square = 1u << 15,

    //adding more Node types here
    Dynamic = 1u << 16,
    Constant = 1u << 17,
    Variable = 1u << 18
};

using utype = std::underlying_type_t<NodeType>;
struct NodeTypes {
    // magic number keeping track of the number of different node types
    static constexpr size_t Count = 19;
    // returns the index of the given type in the NodeType enum
    static size_t GetIndex(NodeType type)
    {
        return std::bitset<Count>(static_cast<utype>(type) - 1).count();
    }
};

inline constexpr NodeType operator&(NodeType lhs, NodeType rhs) { return static_cast<NodeType>(static_cast<utype>(lhs) & static_cast<utype>(rhs)); }
inline constexpr NodeType operator|(NodeType lhs, NodeType rhs) { return static_cast<NodeType>(static_cast<utype>(lhs) | static_cast<utype>(rhs)); }
inline constexpr NodeType operator^(NodeType lhs, NodeType rhs) { return static_cast<NodeType>(static_cast<utype>(lhs) ^ static_cast<utype>(rhs)); }
inline constexpr NodeType operator~(NodeType lhs) { return static_cast<NodeType>(~static_cast<utype>(lhs)); }
inline NodeType& operator&=(NodeType& lhs, NodeType rhs)
{
    lhs = lhs & rhs;
    return lhs;
}
inline NodeType& operator|=(NodeType& lhs, NodeType rhs)
{
    lhs = lhs | rhs;
    return lhs;
}
inline NodeType& operator^=(NodeType& lhs, NodeType rhs)
{
    lhs = lhs ^ rhs;
    return lhs;
}

namespace {
    std::array<std::string, NodeTypes::Count> nodeNames = { "add", "mult", "sub", "div", "aq", "pow", "log", "exp", "sin", "cos", "tan", "tanh", "sqrt", "cbrt", "ln", "square","dyn", "constant", "variable" };
}

struct Node {
    Operon::Hash HashValue;
    Operon::Hash CalculatedHashValue; // for arithmetic terminal nodes whose hash value depends on their children
    Operon::Scalar Value; // value for constants or weighting factor for variables
    uint16_t Arity; // 0-65535
    uint16_t Length; // 0-65535
    uint16_t Depth; // 0-65535
    uint16_t Level;
    uint16_t Parent; // index of parent node
    NodeType Type;
    bool IsEnabled;

    Node() = default; 

    explicit Node(NodeType type) noexcept
        : Node(type, static_cast<Operon::Hash>(type))
    {
    }
    explicit Node(NodeType type, Operon::Hash hashValue) noexcept
        : HashValue(hashValue)
        , CalculatedHashValue(hashValue)
        , Type(type)
    {
        Arity = 0;
        if (Type < NodeType::Log) // Add, Mul, Sub, Div, Aq, Pow
        {
            Arity = 2;
        } else if (Type < NodeType::Constant) // Log, Exp, Sin, Cos, Tan, Tanh, Sqrt, Cbrt, Square
        {
            Arity = 1;
        }
        Length = Arity;

        IsEnabled = true;

        Value = 1.;
    }

    const std::string& Name() const noexcept { return nodeNames[NodeTypes::GetIndex(Type)]; }

    // comparison operators
    inline bool operator==(const Node& rhs) const noexcept
    {
        return CalculatedHashValue == rhs.CalculatedHashValue;
    }

    inline bool operator!=(const Node& rhs) const noexcept
    {
        return !((*this) == rhs);
    }

    inline bool operator<(const Node& rhs) const noexcept
    {
        return HashValue == rhs.HashValue ? CalculatedHashValue < rhs.CalculatedHashValue : HashValue < rhs.HashValue;
    }

    inline bool operator<=(const Node& rhs) const noexcept
    {
        return ((*this) == rhs || (*this) < rhs);
    }

    inline bool operator>(const Node& rhs) const noexcept
    {
        return !((*this) <= rhs);
    }

    inline bool operator>=(const Node& rhs) const noexcept
    {
        return !((*this) < rhs);
    }
    
    inline constexpr bool IsLeaf() const noexcept { return Arity == 0; }
    inline constexpr bool IsCommutative() const noexcept { return Type < NodeType::Sub; }

    template <NodeType... T>
    inline bool Is() const { return ((Type == T) || ...); }

    inline bool IsConstant() const { return Is<NodeType::Constant>(); }
    inline bool IsVariable() const { return Is<NodeType::Variable>(); }
    inline bool IsAddition() const { return Is<NodeType::Add>(); }
    inline bool IsSubtraction() const { return Is<NodeType::Sub>(); }
    inline bool IsMultiplication() const { return Is<NodeType::Mul>(); }
    inline bool IsDivision() const { return Is<NodeType::Div>(); }
    inline bool IsAq() const { return Is<NodeType::Aq>(); }
    inline bool IsPow() const { return Is<NodeType::Pow>(); }
    inline bool IsExp() const { return Is<NodeType::Exp>(); }
    inline bool IsLog() const { return Is<NodeType::Log>(); }
    inline bool IsSin() const { return Is<NodeType::Sin>(); }
    inline bool IsCos() const { return Is<NodeType::Cos>(); }
    inline bool IsTan() const { return Is<NodeType::Tan>(); }
    inline bool IsTanh() const { return Is<NodeType::Tanh>(); }
    inline bool IsSquareRoot() const { return Is<NodeType::Sqrt>(); }
    inline bool IsCubeRoot() const { return Is<NodeType::Cbrt>(); }
    inline bool IsSquare() const { return Is<NodeType::Square>(); }
    inline bool IsDynamic() const { return Is<NodeType::Dynamic>(); }
};
}
#endif // NODE_H

