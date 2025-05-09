#ifndef _AUTOMATICDIFFERENTIATION_HPP_
#define _AUTOMATICDIFFERENTIATION_HPP_
#include <cstddef>
#include <cassert>
#include <vector>
#include <cmath>

/**
* @file   AutomaticDifferentiation.hpp
* @brief  Forward mode automatic differentiation of "N-th order derivative of a single-valued function"
* @author Masaru Sawata <masaru.sawata@tamu.edu>
*/

// 
// This class provides the object that automatically calculates until the N-th order derivatives.
// It has a vector and a pointer as member variables.
// 
// <Initialization>
// var = [x(initial value), 1, 0, 0, 0, ..., 0]
//              ^
//              |
//             m_x
// 
// <Calculation (Operator Overload)>
// This Automatic Differentiation program is based on the metaprograming tequnique, and Operator Overload is implemented recursively.
// 
// For example, multiplication is defined as follows:
// friend AD_single<N> operator*(const AD_single<N>& f, const AD_single<N>& g) { return AD_single<N>(f.x() * g.x(), f.dx() * g.xder() + f.xder() * g.dx()); }
// where f.dx() and f.xder() are the objects of AD_single<N-1>.
// 
// f.dx() and f.xder() are instantited without the std::vector<double> allocation to reduce the unnecessary memory allocation.
// They have a pointer to the element of the var of AD_single<N>.
// 
// <Data Access>
// We can access the function value and its derivatives as follows:
// 
// AD_single<10> x(2.3);
// AD_single<10> f = x*x - 2.0*x + cos(x);
// 
// std::cout << f(0) << std::endl; // Function value
// std::cout << f(1) << std::endl; // 1st order derivative
// std::cout << f(2) << std::endl; // 2nd order derivative
// std::cout << f(3) << std::endl; // 3rd order derivative
// std::cout << f(4) << std::endl; // 4th order derivative
// std::cout << f(5) << std::endl; // 5th order derivative
// std::cout << f(6) << std::endl; // 6th order derivative
// std::cout << f(7) << std::endl; // 7th order derivative
// std::cout << f(8) << std::endl; // 8th order derivative
// std::cout << f(9) << std::endl; // 9th order derivative
// std::cout << f(10) << std::endl; // 10th order derivative
//
template<std::size_t N>
class AD_single {
    std::vector<double> var;
    double* m_x;
public:
    AD_single(const double& _x) : var(N + 1, 0.0) {
        var[0] = _x;
        var[1] = 1.0;
        m_x = var.data();
    }
    AD_single(const double& _x, const AD_single<N - 1>& _dx) : var(N + 1, 0.0) {
        var[0] = _x;
        for (std::size_t i = 1; i < N + 1; ++i) {
            var[i] = _dx(i - 1);
        }
        m_x = var.data();
    }

    AD_single(double* _x) : m_x(_x) {}

    double operator() (std::size_t i)const { return m_x[i]; }
    double x() const { return m_x[0]; }
    AD_single<N - 1> xder() const { return AD_single<N - 1>(m_x); }
    AD_single<N - 1> dx() const { return AD_single<N - 1>(m_x + 1); }

    // Addition
    friend AD_single<N> operator+(const AD_single<N>& f, const AD_single<N>& g) { return AD_single<N>(f.x() + g.x(), f.dx() + g.dx()); }
    friend AD_single<N> operator+(const double& v, const AD_single<N>& f) { return AD_single<N>(v + f.x(), f.dx()); }
    friend AD_single<N> operator+(const AD_single<N>& f, const double& v) { return AD_single<N>(f.x() + v, f.dx()); }

    // Multiplication
    friend AD_single<N> operator*(const AD_single<N>& f, const AD_single<N>& g) { return AD_single<N>(f.x() * g.x(), f.dx() * g.xder() + f.xder() * g.dx()); }
    friend AD_single<N> operator*(const double& v, const AD_single<N>& f) { return AD_single<N>(v * f.x(), v * f.dx()); }
    friend AD_single<N> operator*(const AD_single<N>& f, const double& v) { return AD_single<N>(f.x() * v, f.dx() * v); }

    // Subtraction
    friend AD_single<N> operator-(const AD_single<N>& f) { return AD_single<N>(-f.x(), -f.dx()); }
    friend AD_single<N> operator-(const AD_single<N>& f, const AD_single<N>& g) { return AD_single<N>(f.x() - g.x(), f.dx() - g.dx()); }
    friend AD_single<N> operator-(const double& v, const AD_single<N>& f) { return AD_single<N>(v - f.x(), -f.dx()); }
    friend AD_single<N> operator-(const AD_single<N>& f, const double& v) { return AD_single<N>(f.x() - v, f.dx()); }

    // Division
    friend AD_single<N> operator/(const AD_single<N>& f, const AD_single<N>& g) { return AD_single<N>(f.x() / g.x(), (f.dx() * g.xder() - f.xder() * g.dx()) / (g.xder() * g.xder())); }
    friend AD_single<N> operator/(const double& v, const AD_single<N>& f) { return AD_single<N>(v / f.x(), -v * f.dx() / (f.xder() * f.xder())); }
    friend AD_single<N> operator/(const AD_single<N>& f, const double& v) { return AD_single<N>(f.x() / v, f.dx() / v); }

    // Other functions
    friend AD_single<N> sin(const AD_single<N>& f) { return AD_single<N>(std::sin(f.x()), f.dx() * cos(f.xder())); }
    friend AD_single<N> cos(const AD_single<N>& f) { return AD_single<N>(std::cos(f.x()), -f.dx() * sin(f.xder())); }
    friend AD_single<N> tan(const AD_single<N>& f) { return AD_single<N>(std::tan(f.x()), f.dx() / (cos(f.xder()) * cos(f.xder()))); }

    friend AD_single<N> exp(const AD_single<N>& f) { return AD_single<N>(std::exp(f.x()), f.dx() * exp(f.xder())); }
    friend AD_single<N> log(const AD_single<N>& f) { return AD_single<N>(std::log(f.x()), f.dx() / f.xder()); }
    friend AD_single<N> log10(const AD_single<N>& f) { return AD_single<N>(log(f) / 2.302585092994046); } // ln(10) = 2.302585092994046

    friend AD_single<N> sqrt(const AD_single<N>& f) { return AD_single<N>(std::sqrt(f.x()), 0.5 * f.dx() / sqrt(f.xder())); }
    friend AD_single<N> cbrt(const AD_single<N>& f) { return AD_single<N>(std::cbrt(f.x()), f.dx() / (cbrt(f.xder() * f.xder()) * 3.0)); }

    friend AD_single<N> pow(const AD_single<N>& f, const AD_single<N>& g) { return AD_single<N>(exp(g * log(f))); }
    friend AD_single<N> pow(const double& v, const AD_single<N>& f) { return AD_single<N>(exp(f * std::log(v))); }
    friend AD_single<N> pow(const AD_single<N>& f, const double& v) { return AD_single<N>(std::pow(f.x(), v), v * f.dx() * std::pow(f.x(), v - 1.0)); }
};

template< >
class AD_single<1> {
    std::vector<double> var;
    double* m_x;
public:
    AD_single(const double& _x) : var(2, 0.0) {
        var[0] = _x;
        var[1] = 1.0;
        m_x = var.data();
    }
    AD_single(const double& _x, const double& _dx) : var(2, 0.0) {
        var[0] = _x;
        var[1] = _dx;
        m_x = var.data();
    }
    AD_single(double* _x) : m_x(_x) {}

    double operator()(std::size_t i) const { return m_x[i]; }
    double x() const { return m_x[0]; }
    double xder() const { return m_x[0]; }
    double dx() const { return m_x[1]; }

    // Addition
    friend AD_single<1> operator+(const AD_single<1>& f, const AD_single<1>& g) { return AD_single<1>(f.x() + g.x(), f.dx() + g.dx()); }
    friend AD_single<1> operator+(const double& v, const AD_single<1>& f) { return AD_single<1>(v + f.x(), f.dx()); }
    friend AD_single<1> operator+(const AD_single<1>& f, const double& v) { return AD_single<1>(f.x() + v, f.dx()); }

    // Multiplication
    friend AD_single<1> operator*(const AD_single<1>& f, const AD_single<1>& g) { return AD_single<1>(f.x() * g.x(), f.dx() * g.xder() + f.xder() * g.dx()); }
    friend AD_single<1> operator*(const double& v, const AD_single<1>& f) { return AD_single<1>(v * f.x(), v * f.dx()); }
    friend AD_single<1> operator*(const AD_single<1>& f, const double& v) { return AD_single<1>(f.x() * v, f.dx() * v); }

    // Subtraction
    friend AD_single<1> operator-(const AD_single<1>& f) { return AD_single<1>(-f.x(), -f.dx()); }
    friend AD_single<1> operator-(const AD_single<1>& f, const AD_single<1>& g) { return AD_single<1>(f.x() - g.x(), f.dx() - g.dx()); }
    friend AD_single<1> operator-(const double& v, const AD_single<1>& f) { return AD_single<1>(v - f.x(), -f.dx()); }
    friend AD_single<1> operator-(const AD_single<1>& f, const double& v) { return AD_single<1>(f.x() - v, f.dx()); }

    // Division
    friend AD_single<1> operator/(const AD_single<1>& f, const AD_single<1>& g) { return AD_single<1>(f.x() / g.x(), (f.dx() * g.xder() - f.xder() * g.dx()) / (g.xder() * g.xder())); }
    friend AD_single<1> operator/(const double& v, const AD_single<1>& f) { return AD_single<1>(v / f.x(), -v * f.dx() / (f.xder() * f.xder())); }
    friend AD_single<1> operator/(const AD_single<1>& f, const double& v) { return AD_single<1>(f.x() / v, f.dx() / v); }

    // Other functions
    friend AD_single<1> sin(const AD_single<1>& f) { return AD_single<1>(std::sin(f.x()), f.dx() * cos(f.xder())); }
    friend AD_single<1> cos(const AD_single<1>& f) { return AD_single<1>(std::cos(f.x()), -f.dx() * sin(f.xder())); }
    friend AD_single<1> tan(const AD_single<1>& f) { return AD_single<1>(std::tan(f.x()), f.dx() / (cos(f.xder()) * cos(f.xder()))); }

    friend AD_single<1> exp(const AD_single<1>& f) { return AD_single<1>(std::exp(f.x()), f.dx() * exp(f.xder())); }
    friend AD_single<1> log(const AD_single<1>& f) { return AD_single<1>(std::log(f.x()), f.dx() / f.xder()); }
    friend AD_single<1> log10(const AD_single<1>& f) { return AD_single<1>(log(f) / 2.302585092994046); } // ln(10) = 2.302585092994046

    friend AD_single<1> sqrt(const AD_single<1>& f) { return AD_single<1>(std::sqrt(f.x()), 0.5 * f.dx() / sqrt(f.xder())); }
    friend AD_single<1> cbrt(const AD_single<1>& f) { return AD_single<1>(std::cbrt(f.x()), f.dx() / (cbrt(f.xder() * f.xder()) * 3.0)); }

    friend AD_single<1> pow(const AD_single<1>& f, const AD_single<1>& g) { return AD_single<1>(exp(g * log(f))); }
    friend AD_single<1> pow(const double& v, const AD_single<1>& f) { return AD_single<1>(exp(f * std::log(v))); }
    friend AD_single<1> pow(const AD_single<1>& f, const double& v) { return AD_single<1>(std::pow(f.x(), v), v * f.dx() * std::pow(f.x(), v - 1.0)); }
};


#endif





