#include <iostream>
#include <string>

#include "bfgs.h"


void print(const std::string& name, double y, const double* const x, int n, int fun_n)
{
	std::cout << name << std::endl;
	std::cout << "fun_n = " << fun_n << std::endl;
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

	// Numerical derivative.
	{
		int fun_n = 0;
		auto f = [&fun_n](const double* const x, int n) -> double
		{
			++fun_n;
			return x[0] * x[0] + x[1] * x[1] + x[0] + 2 * x[1];
		};
		const int n = 2;
		double x[n] = {0.0, 0.0};
		double y = bfgs.find_min(f, x, n);
		print("x1^2 + x2^2 + x1 + 2 * x2 (numerical)", y, x, n, fun_n);
	}

	// Analytic derivative.
	{
		int fun_n = 0;
		auto f = [&fun_n](const double* const x, double* const g, int n) -> double
		{
			++fun_n;
			g[0] = 2 * x[0] + 1;
			g[1] = 2 * x[1] + 2;
			return x[0] * x[0] + x[1] * x[1] + x[0] + 2 * x[1];
		};
		const int n = 2;
		double x[n] = {0.0, 0.0};
		double y = bfgs.find_min(f, x, n);
		print("x1^2 + x2^2 + x1 + 2 * x2 (analytic)", y, x, n, fun_n);
	}

	return 0;
}
