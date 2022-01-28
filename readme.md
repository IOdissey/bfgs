# BFGS


### About
This is a c++ implementation of the BFGS algorithm.


### Features
* header only (only one file)
* no dependencies


### Usage

Include header
```
#include "bfgs.h"
```

Create and configure BFGS class
```
BFGS bfgs;
bfgs.set_grad_eps(1e-8);
bfgs.set_stop_grad_eps(1e-7);
bfgs.set_stop_step_eps(1e-7);
bfgs.set_max_iter(1000);
```

Create the problem and solve it
```
auto f = [](const double* const x, int n) -> double
{
    return x[0] * x[0] + x[1] * x[1] + x[0] + 2 * x[1];
};
const int n = 2;
double x[n] = {0.0, 0.0};
double y = bfgs.find_min(f, x, n);
```

You can use analytic derivatives
```
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

### Example
* [simple](example/simple)
* [compare with dlib](example/dlib)


### Options

Epsilon for calculating gradients. It is also used for some other checks.<br/>
Default: 1e-8.
```
bfgs.set_grad_eps(1e-8);
```

---
Stop criterion. Gradient Norm.<br/>
Default: 1e-7.
```
bfgs.set_stop_grad_eps(1e-7);
```

---
Stop criterion. The function change size between iterations.<br/>
Default: 1e-7.
```
bfgs.set_stop_step_eps(1e-7);
```

---
Stop criterion. The maximum number of iterations of the algorithm.<br/>
Default: 1000.
```
bfgs.set_max_iter(1000);
```

---
Wolfe criterion constants (0.0 < c1 < c2 < 1.0).<br/>
Default: 0.01 and 0.9.
```
bfgs.set_c1_c2(0.01, 0.9);
```

---
The maximum number of iterations of the line search.<br/>
Default: 100.
```
bfgs.set_line_max_iter(100);
```

---
Use central difference or forward difference for the gradient.<br/>
Default: true.
```
bfgs.set_central_diff(true);
```

---
Use central difference or forward difference for the line serach.<br/>
Maybe we don't want too much precision here.<br/>
You can try it to reduce the number of function calls (set to false).<br/>
Default: true.
```
bfgs.set_line_central_diff(false);
```

---
Estimated minimum of the function.<br/>
Used for stop criterion if<br/>
fk - grad_eps < min_f<br/>
Also used for step_tweak option.<br/>
Default: inf (not use).
```
bfgs.set_min_f(-1.0);
```

---
Experimental setting of the first step of the linear search.<br/>
If set, then the first step of the linear search will be:<br/>
a = min(1, step_tweak * (f_min - f) / drad(f))<br/>
Default: 0.0 (not use step_tweak if <= 0.0)
```
bfgs.set_step_tweak(2.0);
```

---
Experimental setting.<br/>
When solving a large number of similar problems, the value from the previous run can be used as the initial approximation of the inverse Hessian.<br/>
Default: false.
```
bfgs.set_reuse_hessian(true);
```

---
If we want to save some memory for a large problem.<br/>
In the usual case, the memory size is proportional to<br/>
n * (n + 4)<br/>
In the case of storing only the upper triangular inverse Hessian matrix, the memory size is proportional to<br/>
n * (n + 9) / 2<br/>
Default: false.<br/>
```
bfgs.set_memory_save(true);
```

---
Use Strong Wolfe conditions.<br/>
Defaul: false.
```
bfgs.set_strong_wolfe(true);
```
