#include <iostream>
#include <string>
#include <chrono>

#define BFGS_AUTO
#include <bfgs/bfgs.h>


void print(const std::string& name, double y, const double* const x, int n, int fun_call)
{
	std::cout << name << std::endl;
	std::cout << "fun_call = " << fun_call << std::endl;
	std::cout << "y = " << y << std::endl;
	for (int i = 0; i < n; ++ i)
		std::cout << "x" << i + 1 << " = " << x[i] << std::endl;
	std::cout << std::endl;
}

int main()
{
	BFGS bfgs;
	bfgs.set_grad_eps(1e-8);
	bfgs.set_stop_grad_eps(1e-7);
	bfgs.set_stop_step_eps(1e-7);
	bfgs.set_max_iter(1000);
	bfgs.set_dval_size(20);

	// Numerical derivative.
	{
		uint32_t fun_call = 0;
		auto f = [&fun_call](const double* const x, uint32_t n) -> double
		{
			++fun_call;
			return x[0] * x[0] + x[1] * x[1] + x[0] + 2 * x[1];
		};
		const uint32_t n = 2;
		double x[n] = {0.0, 0.0};
		double y = bfgs.find_min_num(f, x, n);
		// or
		// double y = bfgs.find_min(f, x, n);
		print("x1^2 + x2^2 + x1 + 2 * x2 (numerical)", y, x, n, fun_call);
	}

	// Analytic derivative.
	{
		uint32_t fun_call = 0;
		auto f = [&fun_call](const double* const x, double* const g, uint32_t n) -> double
		{
			++fun_call;
			g[0] = 2 * x[0] + 1;
			g[1] = 2 * x[1] + 2;
			return x[0] * x[0] + x[1] * x[1] + x[0] + 2 * x[1];
		};
		const uint32_t n = 2;
		double x[n] = {0.0, 0.0};
		double y = bfgs.find_min_grad(f, x, n);
		// or
		// double y = bfgs.find_min(f, x, n);
		print("x1^2 + x2^2 + x1 + 2 * x2 (analytic)", y, x, n, fun_call);
	}

#ifdef BFGS_AUTO
	// Automatic derivative (dynamic dimension).
	{
		uint32_t fun_call = 0;
		auto f = [&fun_call](const DVal<0>* const x, uint32_t n) -> DVal<0>
		{
			++fun_call;
			return x[0] * x[0] + x[1] * x[1] + x[0] + 2 * x[1];
		};
		const uint32_t n = 2;
		double x[n] = {0.0, 0.0};
		double y = bfgs.find_min_auto(f, x, n);
		print("x1^2 + x2^2 + x1 + 2 * x2 (dynamic dimension)", y, x, n, fun_call);
	}

	// Automatic derivative (fixed dimension).
	{
		uint32_t fun_call = 0;
		auto f = [&fun_call](const DVal<2>* const x, uint32_t n) -> DVal<2>
		{
			++fun_call;
			return x[0] * x[0] + x[1] * x[1] + x[0] + 2 * x[1];
		};
		const uint32_t n = 2;
		double x[n] = {0.0, 0.0};
		double y = bfgs.find_min_auto<2>(f, x, n);
		print("x1^2 + x2^2 + x1 + 2 * x2 (fixed dimension)", y, x, n, fun_call);
	}
#endif

	return 0;
}