#include <iostream>
#include <string>
#include <chrono>

#include <bfgs/var/bfgs.h>
#include <bfgs/fix/bfgs.h>


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
	bfgs::var::BFGS bfgs_var;
	bfgs_var.set_grad_eps(1e-8);
	bfgs_var.set_stop_grad_eps(1e-7);
	bfgs_var.set_stop_step_eps(1e-7);
	bfgs_var.set_max_iter(1000);
	bfgs_var.set_dval_size(20);

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
		double y = bfgs_var.find_min(f, x, n);
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
		double y = bfgs_var.find_min(f, x, n);
		print("x1^2 + x2^2 + x1 + 2 * x2 (analytic)", y, x, n, fun_call);
	}

	// Automatic derivative.
	{
		uint32_t fun_call = 0;
		auto f = [&fun_call](const bfgs::var::DVal* const x, uint32_t n) -> bfgs::var::DVal
		{
			++fun_call;
			return x[0] * x[0] + x[1] * x[1] + x[0] + 2 * x[1];
		};
		const uint32_t n = 2;
		double x[n] = {0.0, 0.0};
		double y = bfgs_var.find_min(f, x, n);
		print("x1^2 + x2^2 + x1 + 2 * x2 (automatic)", y, x, n, fun_call);
	}

	// Automatic derivative (static version).
	{
		const uint32_t n = 2;
		bfgs::fix::BFGS<2> bfgs_fix;
		bfgs_fix.set_grad_eps(1e-8);
		bfgs_fix.set_stop_grad_eps(1e-7);
		bfgs_fix.set_stop_step_eps(1e-7);
		bfgs_fix.set_max_iter(1000);

		uint32_t fun_call = 0;
		auto f = [&fun_call](const bfgs::fix::DVal<2>* const x, uint32_t n) -> bfgs::fix::DVal<2>
		{
			++fun_call;
			return x[0] * x[0] + x[1] * x[1] + x[0] + 2 * x[1];
		};
		double x[n] = {0.0, 0.0};
		double y = bfgs_fix.find_min(f, x, n);
		print("x1^2 + x2^2 + x1 + 2 * x2 (automatic static)", y, x, n, fun_call);
	}

	return 0;
}
