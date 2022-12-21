# BFGS


## About
This is a c++ implementation of the BFGS algorithm.


## Features
* header only
* no dependencies
* automatic derivative possible
* lbfgs version possible


### Note
The automatic derivative is an experimental option. Often this does not give advantages over numerical derivatives.


## Usage

Include header
```cpp
#include <bfgs/bfgs.h>
```

Create and configure BFGS class
```cpp
BFGS bfgs;
bfgs.set_grad_eps(1e-8);
bfgs.set_stop_grad_eps(1e-7);
bfgs.set_stop_step_eps(1e-7);
bfgs.set_max_iter(1000);
```

Create the problem and solve it
```cpp
auto f = [](const double* const x, int n) -> double
{
    return x[0] * x[0] + x[1] * x[1] + x[0] + 2 * x[1];
};
const int n = 2;
double x[n] = {0.0, 0.0};
double y = bfgs.find_min(f, x, n);
```

You can use analytic derivatives
```cpp
auto f = [](const double* const x, double* const g, int n) -> double
{
    g[0] = 2 * x[0] + 1;
    g[1] = 2 * x[1] + 2;
    return x[0] * x[0] + x[1] * x[1] + x[0] + 2 * x[1];
};
const int n = 2;
double x[n] = {0.0, 0.0};
double y = bfgs.find_min(f, x, n);
```

It is possible to use automatic derivatives. It can be enabled with a macro `BFGS_AUTO` before include.
```cpp
#define BFGS_AUTO
#include <bfgs/bfgs.h>
```

You can use automatic derivatives with fixed dimension.
```cpp
auto f = [](const DVal<2>* const x, uint32_t n) -> DVal<2>
{
    return x[0] * x[0] + x[1] * x[1] + x[0] + 2 * x[1];
};
const int n = 2;
double x[n] = {0.0, 0.0};
double y = bfgs.find_min_auto<2>(f, x, n);
```

You can use automatic derivatives with dynamic dimension.
```cpp
auto f = [](const DVal<0>* const x, uint32_t n) -> DVal<0>
{
    return x[0] * x[0] + x[1] * x[1] + x[0] + 2 * x[1];
};
const int n = 2;
double x[n] = {0.0, 0.0};
double y = bfgs.find_min_auto(f, x, n);
```


## Example
* [simple examples](example/simple)
* [compare with dlib](example/dlib)
* [demonstration of automatic differentiation](example/autodiff)


## Options

> **grad_eps**<br/>
Epsilon for calculating gradients. It is also used for some other checks.<br/>
Default: `1e-8`.
> ```
> bfgs.set_grad_eps(1e-8);
> ```

> **stop_grad_eps**<br/>
Stop criterion. Gradient Norm.<br/>
Default: `1e-7`.
> ```
> bfgs.set_stop_grad_eps(1e-7);
> ```

> **stop_step_eps**<br/>
Stop criterion. The function change size between iterations.<br/>
Default: `1e-7`.
> ```
> bfgs.set_stop_step_eps(1e-7);
> ```

> **max_iter**<br/>
Stop criterion. The maximum number of iterations of the algorithm.<br/>
Default: `1000`.
> ```
> bfgs.set_max_iter(1000);
> ```

> **c1 and c2**<br/>
Wolfe criterion constants (`0.0 < c1 < c2 < 1.0`).<br/>
Default: `0.01` and `0.9`.
> ```
> bfgs.set_c1_c2(0.01, 0.9);
> ```

> **line_max_iter**<br/>
The maximum number of iterations of the line search.<br/>
Default: `100`.
> ```
> bfgs.set_line_max_iter(100);
> ```

> **central_diff**<br/>
Only for the case of a numerical derivative.<br/>
Use central difference or forward difference for the gradient.<br/>
Default: `true`.
> ```
> bfgs.set_central_diff(true);
> ```

> **line_central_diff**<br/>
Only for the case of a numerical derivative.<br/>
Use central difference or forward difference for the line serach.<br/>
Maybe we don't want too much precision here.<br/>
You can try it to reduce the number of function calls (set to `false`).<br/>
Default: `true`.
> ```
> bfgs.set_line_central_diff(false);
> ```

> **min_f**<br/>
Estimated minimum of the function.<br/>
Used for stop criterion if<br/>
`fk - grad_eps < min_f`<br/>
Also used for step_tweak option.<br/>
Default: `inf` (not use).
> ```
> bfgs.set_min_f(-1.0);
> ```

> **step_tweak**<br/>
Experimental setting of the first step of the linear search.<br/>
If set, then the first step of the linear search will be:<br/>
`a = min(1, step_tweak * (f_min - f) / grad(f))`<br/>
Default: `0.0` (not use step_tweak if <= `0.0`)
> ```
> bfgs.set_step_tweak(2.0);
> ```

> **reuse_hessian**<br/>
Experimental setting.<br/>
When solving a large number of similar problems, the value from the previous run can be used as the initial approximation of the inverse Hessian.<br/>
Default: `false`.
> ```
> bfgs.set_reuse_hessian(true);
> ```

> **memory_save**<br/>
If we want to save some memory for a large problem.<br/>
In the usual case, the memory size is proportional to:<br/>
`n * (n + 4)`<br/>
In the case of storing only the upper triangular inverse Hessian matrix, the memory size is proportional to:<br/>
`n * (n + 9) / 2`<br/>
Not too important for the case of automatic differentiation.<br/>
Default: `false`.
> ```
> bfgs.set_memory_save(true);
> ```

> **select_hessian**<br/>
Selection of the initial approximation of the inverse hessian.<br/>
Default: `false`.
> ```
> bfgs.set_select_hessian(true);
> ```

> **strong_wolfe**<br/>
Use the strong Wolfe conditions (`true`).<br/>
Use the normal Wolfe condition (`false`).<br/>
Defaul: `true`.
> ```
> bfgs.set_strong_wolfe(true);
> ```

> **dval_size**<br/>
Specifies the amount of memory for automatic derivative with dynamic dimension.<br/>
If set to `0` then memory is not reserved in advance (may be slower).<br/>
Defaul: `100`.
> ```
> bfgs.set_dval_size(100);
> ```

> **line_force_num**<br/>
Force use of numerical derivative in linear search for the case of an analytic derivative.<br/>
In the case when the gradient is not needed when calling the function, the gradient pointer will be `nullptr`.<br/>
Defaul: `false`.
> ```
> bfgs.set_line_force_num(true);
> ```

> **lbfgs_m**<br/>
History size for lbfgs version. If `0` then the regular version is used.<br/>
The memory size is proportional to:<br/>
`2 * (n * m + n + m)`.<br/>
Defaul: `0`.
> ```
> bfgs.set_lbfgs_m(0);
> ```