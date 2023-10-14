// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_INTERPRETER_FUNCTIONS_HPP
#define OPERON_INTERPRETER_FUNCTIONS_HPP

#include "core/node.hpp"

namespace Operon
{
    // addition up to 5 arguments
    template<Operon::NodeType N = NodeType::Add>
    struct Function 
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a; }

        template<typename T, typename... Tn>
        inline void operator()(T r, T a1, Tn... an) { r = a1 + (an + ...); }
    };

    template<>
    struct Function<NodeType::Sub>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = -a; }

        template<typename T, typename... Tn>
        inline void operator()(T r, T a1, Tn... an) { r = a1 - (an + ...); }
    };

    template<>
    struct Function<NodeType::Mul>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a; }

        template<typename T, typename... Tn>
        inline void operator()(T r, T a1, Tn... an) { r = a1 * (an * ...); }
    };

    template<>
    struct Function<NodeType::Div>
    {
        template<typename T>
        inline void operator()(T r, T a) { 
                        //std::cout<<"in div operator inverse "<<std::endl;

            r = a.inverse(); }

        template<typename T, typename... Tn>
        inline void operator()(T r, T a1, Tn... an) { 
            
            //std::cout<<" inverse contin";
            auto divisor = (an * ...);
            if(divisor.isZero()){
                r = (divisor + (typename T::Scalar{1.0}))/(divisor + (typename T::Scalar{1.0}));
        }
            else{
                r = a1 / divisor; }
        }
    };

    // continuations for n-ary functions (add, sub, mul, div)
    template<NodeType N = NodeType::Add>
    struct ContinuedFunction
    {
        template<typename T>
        inline void operator()(T r, T a) { r += a; }

        template<typename T, typename... Ts>
        inline void operator()(T r, T a1, Ts... an) { r += a1 + (an + ...); }
    };

    template<>
    struct ContinuedFunction<NodeType::Sub>
    {
        template<typename T>
        inline void operator()(T r, T a) { r -= a; }

        template<typename T, typename... Ts>
        inline void operator()(T r, T a1, Ts... an) { r -= a1 + (an + ...); }
    };

    template<>
    struct ContinuedFunction<NodeType::Mul>
    {
        template<typename T>
        inline void operator()(T r, T a) { r *= a; }

        template<typename T, typename... Ts>
        inline void operator()(T r, T a1, Ts... an) { r *= a1 * (an * ...); }
    };

    template<>
    struct ContinuedFunction<NodeType::Div>
    {
        template<typename T>
        inline void operator()(T r, T t) { 
            
            //if(t.value() == (typename T::Scalar{0})){
            //std::cout<<"in div operator "<<std::endl;

            if(t.isZero()) {
                r = (t + (typename T::Scalar{1.0}))/(t + (typename T::Scalar{1.0}));
            }


            //  r = (t.isZero()) ?  (t + (typename T::Scalar{1.0}))/(t + (typename T::Scalar{1.0})) : r/t; 
             else{
                 r /= t;
             }

        }

        template<typename T, typename... Ts>
        inline void operator()(T r, T a1, Ts... an) { 
                    //    std::cout<<"in div operator contin"<<std::endl;

            r /= a1 * (an * ...); }
    };

    // binary and unary functions
    template<>
    struct Function<NodeType::Aq>
    {
        template<typename T>
        inline void operator()(T r, T a1, T a2) { r = a1 / (typename T::Scalar{1.0} + a2.square()).sqrt(); };
    };

    template<>
    struct Function<NodeType::Pow>
    {
        template<typename T>
        inline void operator()(T r, T a1, T a2) { r = a1.pow(a2); };
    };

    template<>
    struct Function<NodeType::Log>
    {
        template<typename T>
        inline void operator()(T r, T a) { 
            
            if (a.isZero()) { r = (typename T::Scalar{0.0}); }
            else { r = a.abs().log(); } }
            // r = a.abs().log(); }
    };

    template<>
    struct Function<NodeType::Exp>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.exp(); }
    };

    template<>
    struct Function<NodeType::Sin>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.sin(); }
    };

    template<>
    struct Function<NodeType::Cos>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.cos(); }
    };

    template<>
    struct Function<NodeType::Tan>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.tan(); }
    };

    template<>
    struct Function<NodeType::Tanh>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.tanh(); }
    };

    template<>
    struct Function<NodeType::Sqrt>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.abs().sqrt(); }
    };

    template<>
    struct Function<NodeType::Cbrt>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.unaryExpr([](auto&& v) { return ceres::cbrt(v); }); }
    };

    template<>
    struct Function<NodeType::Square>
    {
        template<typename T>
        inline void operator()(T r, T a) { r = a.square(); }
    };




    template<>
    struct Function<NodeType::Dynamic>
    {
        template<typename T>
        inline void operator()(T, T) { /* do nothing */ }
    };




}

#endif
