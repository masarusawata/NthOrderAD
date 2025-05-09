# Nth-Order AD
The current code has redundant memory allocation and deallocation, but it can compute the N-th order derivative.

It implements forward-mode automatic differentiation recursively using a metaprogramming technique.

You can use it as follows:

```cpp
#include "AutomaticDifferentiation.hpp"
AD_single<10> x(2.3);
AD_single<10> f = x*x - 2.0*x + cos(x);

std::cout << f(0) << std::endl; // Function value
std::cout << f(1) << std::endl; // 1st order derivative
std::cout << f(2) << std::endl; // 2nd order derivative
std::cout << f(3) << std::endl; // 3rd order derivative
std::cout << f(4) << std::endl; // 4th order derivative
std::cout << f(5) << std::endl; // 5th order derivative
std::cout << f(6) << std::endl; // 6th order derivative
std::cout << f(7) << std::endl; // 7th order derivative
std::cout << f(8) << std::endl; // 8th order derivative
std::cout << f(9) << std::endl; // 9th order derivative
std::cout << f(10) << std::endl; // 10th order derivative
```