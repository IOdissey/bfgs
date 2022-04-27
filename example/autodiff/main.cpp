#include <iostream>
#include <string>
#include <chrono>
#include <functional>

#include <bfgs/dval.h>


template <typename T>
T fun(const T* const x, int n)
{
	auto res = x[0] * x[1] * x[2] * x[3] * x[4];
	for (int i = 0; i < 10000; ++i)
	{
		res += 1.1;
		res -= 1.2;
		res *= 1.3;
		res /= 1.4;
		res += x[1];
		res -= x[2];
		res *= x[3];
		res /= x[4];
		res += x[1] + 1.1;
		res += x[2] - 1.2;
		res += x[3] * 1.3;
		res += x[4] / 1.4;
		res += 1.1 + x[1];
		res += 1.2 - x[2];
		res += 1.3 * x[3];
		res += 1.4 / x[4];
		res += x[0] + x[1];
		res += x[0] - x[2];
		res += x[0] * x[3];
		res += x[0] / x[4];
		res += std::sqrt(x[1]);
		res += std::cbrt(x[2]);
		res += std::pow(x[3], 2.5);
		res += std::pow(2.5, x[4]);
		res += std::exp(x[1]);
		res += std::log(x[2]);
		res += std::log10(x[3]);
		res += std::log2(x[4]);
		res += std::abs(x[0] * x[1] * x[2] * x[3] * x[4]);
		res += std::abs(-x[0] * x[1] * x[2] * x[3] * x[4]);
		res += std::sin(x[1]);
		res += std::cos(x[2]);
		res += std::tan(x[3]);
		res += std::asin(x[1] / 5.0);
		res += std::acos(x[2] / 5.0);
		res += std::atan(x[3] / 5.0);
	}
	return res;
}

int main()
{
	std::cout.precision(9);

	// Numerical derivative.
	std::cout << "Numerical derivative:" << std::endl;
	// for (int j = 0; j < 5; ++j)
	{
		const int n = 5;
		double x_arr[n];
		for (int i = 0; i < n; ++i)
			x_arr[i] = 1.0 + i;
		double g_arr[n];

		auto beg = std::chrono::steady_clock::now();
		double y = fun(x_arr, n);
		const double eps = 1e-9;
		for (int i = 0; i < n; ++i)
		{
			const double x = x_arr[i];
			x_arr[i] = x + eps;
			const double yp = fun(x_arr, n);
			x_arr[i] = x - eps;
			const double ym = fun(x_arr, n);
			x_arr[i] = x;
			g_arr[i] = (yp - ym) / (2.0 * eps);

		}
		auto end = std::chrono::steady_clock::now();
		double dt = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count() * 1e-6;

		std::cout << "dt = " << dt << " ms" << std::endl;
		std::cout << y << " (";
		for (int i = 0; i < n; ++i)
		{
			if (i > 0)
				std::cout << ", ";
			std::cout << g_arr[i];
		}
		std::cout << ")" << std::endl;
		std::cout << std::endl;
	}

	// Automatic derivative (dinamic).
	std::cout << "Automatic derivative (dinamic): " << std::endl;
	// for (int j = 0; j < 5; ++j)
	{
		const int n = 5;
		Memory mem(n, 100);
		DVal<0> x_arr[n];
		for (int i = 0; i < n; ++i)
			x_arr[i].set(1.0 + i, i, &mem);

		auto beg = std::chrono::steady_clock::now();
		auto res = fun(x_arr, n);
		auto end = std::chrono::steady_clock::now();
		double dt = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count() * 1e-6;

		std::cout << "dt = " << dt << " ms" << std::endl;
		std::cout << res << std::endl;
		std::cout << std::endl;
	}

	// Automatic derivative (fixed).
	std::cout << "Automatic derivative (fixed): " << std::endl;
	// for (int j = 0; j < 5; ++j)
	{
		const int n = 5;
		DVal<5> x_arr[n];
		for (int i = 0; i < n; ++i)
			x_arr[i].set(1.0 + i, i);

		auto beg = std::chrono::steady_clock::now();
		auto res = fun(x_arr, n);
		auto end = std::chrono::steady_clock::now();
		double dt = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count() * 1e-6;

		std::cout << "dt = " << dt << " ms" << std::endl;
		std::cout << res << std::endl;
		std::cout << std::endl;
	}

	return 0;
}