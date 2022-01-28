#include <dlib/optimization.h>

#include "bfgs.h"

int _fun_n = 0;

// Rosenbrock's function.
class FRosen
{
private:
	const double _p1;

	double _f(double x1, double x2) const
	{
		++_fun_n;
		const double v1 = x2 - x1 * x1;
		const double v2 = _p1 - x1;
		return 100.0 * v1 * v1 + v2 * v2 + 1.0;
	}

public:
	FRosen(double p1 = 1.0):
		_p1(p1)
	{
	}

	double operator()(const dlib::matrix<double,0,1>& m) const
	{
		return _f(m(0), m(1));
	}

	double operator()(const double* const x, int n) const
	{
		return _f(x[0], x[1]);
	}
};

// Simple function.
class F1
{
private:
	const double _p1;

	double _f(double x1, double x2) const
	{
		++_fun_n;
		return _p1 * x1 * x1 + x2 * x2 + x1 + x2;
	}

public:
	F1(double p1 = 1.0):
		_p1(p1)
	{
	}

	double operator()(const dlib::matrix<double,0,1>& m) const
	{
		return _f(m(0), m(1));
	}

	double operator()(const double* const x, int n) const
	{
		return _f(x[0], x[1]);
	}
};

template <typename fun>
void test(const fun& f)
{
	_fun_n = 0;
	dlib::matrix<double,0,1> dlib_point = {-1.0, 1.0};
	double dlib_y = dlib::find_min_using_approximate_derivatives(
			dlib::bfgs_search_strategy(),
			// dlib::gradient_norm_stop_strategy(1e-7, 1000),
			dlib::objective_delta_stop_strategy(1e-7, 1000),
			f,
			dlib_point,
			-1,
			1e-8);
	std::cout << "\tdlib" << std::endl;
	std::cout << "fun_n: " << _fun_n << "\nsolution: " << dlib_y  << std::endl <<  dlib_point;

	_fun_n = 0;
	const int n = 2;
	double bfgs_point[n] = {-1.0, -1.0};
	BFGS bfgs;
	bfgs.set_grad_eps(1e-8);
	bfgs.set_stop_grad_eps(1e-7);
	bfgs.set_stop_step_eps(1e-7);
	bfgs.set_max_iter(1000);
	bfgs.set_line_central_diff(false);
	double bfgs_y = bfgs.find_min(f, bfgs_point, n);
	std::cout << "\tbfgs" << std::endl;
	std::cout << "fun_n: " << _fun_n << "\nsolution: " << dlib_y << std::endl;
	for (int i = 0; i < n; ++i)
		std::cout << bfgs_point[i] << std::endl;

	std::cout << std::endl;
}

int main()
{
	std::cout.precision(9);

	for (int i = 1; i <= 100; i *= 10)
		test(FRosen(i));
	for (int i = 1; i <= 100; i *= 10)
		test(F1(i));

	return 0;
}