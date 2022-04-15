// Copyright © 2022 Alexander Abramenkov. All rights reserved.
// Licensed under the Apache License, Version 2.0.
// https://github.com/IOdissey/bfgs

#pragma once

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>

#include "dval.h"


namespace bfgs::fix
{
	// Version with static dimension and automatic differentiation.
	template <uint32_t N>
	class BFGS
	{
	private:
		double _g[N];                   // The gradient value.
		double _dg[N];                  // The gradient increment.
		double _p[N];                   // Search direction.
		double _xi[N];                  // New point.
		double _h[N * N];               // Approximation of the inverse Hessian.
		DVal<N> _dval[N];               // Automatic derivative.

		double _min_f = std::numeric_limits<double>::infinity();
		double _stop_grad_eps = 1e-7;   // Stop criteria.
		double _stop_step_eps = 1e-9;   // Stop criteria.
		double _eps = 1e-8;             // Increment for calculating the derivative and another. TODO. Use different values.
		uint32_t _max_iter = 1000;      // Maximum number of algorithm iterations.
		double _c1 = 0.01;              // Constant for checking the Wolfe condition.
		double _c2 = 0.9;               // Constant for checking the Wolfe condition.
		uint32_t _line_iter_max = 100;  // Maximum number of linear search iterations.
		const double _db = 2.0;         // Upper bound increment multiplier.
		double _step_tweak = 0.0;       // Initial step value tweak (if _step_tweak > 0.0).
		bool _reuse_hessian = false;    // Reuse hessian from last time.
		bool _use_strong_wolfe = true;  // Use Strong Wolfe conditions.
		
		// Diagonal matrix.
		void _diag(double* m, const double val = 1.0) const
		{
			constexpr uint32_t nn = N * N;
			constexpr uint32_t step = N + 1;
			std::fill_n(m, nn, 0.0);
			for (uint32_t i = 0; i < nn; i += step)
				m[i] = val;
		}

		// L2 norm.
		double _norm(const double* v) const
		{
			double r = 0.0;
			for (uint32_t i = 0; i < N; ++i)
				r += v[i] * v[i];
			return std::sqrt(r);
		}

		// r = m * v.
		void _mull_m_v(const double* m, const double* v, double* r) const
		{
			for (uint32_t i = 0; i < N; ++i)
			{
				r[i] = 0.0;
				for (uint32_t j = 0; j < N; ++j, ++m)
					r[i] += *m * v[j];
			}
		}

		// r = v1 + a * v2.
		void _add_v_av(const double* v1, const double* v2, const double a, double* r) const
		{
			for (uint32_t i = 0; i < N; ++i)
				r[i] = v1[i] + a * v2[i];
		}

		// r = v1^T * v2.
		double _mull_v_v(const double* v1, const double* v2) const
		{
			double r = 0.0;
			for (uint32_t i = 0; i < N; ++i)
				r += v1[i] * v2[i];
			return r;
		}

		// r = v.
		inline void _copy_v(const double* v, double* const r) const
		{
			std::memcpy(r, v, N * sizeof(double));
		}

		// r = v1 - v2.
		inline void _sub_v_v(const double* const v1, const double* const v2, double* const r) const
		{
			for (uint32_t i = 0; i < N; ++i)
				r[i] = v1[i] - v2[i];
		}

		// Cubic approximation for finding the minimum.
		// There is always exists the minimum if ga < 0 and gb > 0.
		// f(x) = a0 + a1 * x + a2 * x^2 + a3 * x^3
		// f(a) = fa
		// f'(a) = da
		// f(b) = fb
		// f'(b) = db
		double _poly(
			const double a, const double b,
			const double fa, const double da,
			const double fb, const double db,
			const double limit_min = 0.0, const double limit_max = 1.0) const
		{
			// Search the minimum for the range [0, 1] and then scale to [a, b].
			// f(0) = fa
			// f'(0) = da
			// f(1) = fb
			// f'(1) = db
			const double& a0 = fa;
			const double& a1 = da;
			const double a2 = 3 * (fb - fa) - 2 * da - db;
			const double a3 = da + db - 2 * (fb - fa);
			// 
			double x_min = std::numeric_limits<double>::max();
			// f'(x) = a1 + 2 * a2 * x + 3 * a3 * x^2 = 0
			if (std::abs(a3) > _eps)
			{
				double d = a2 * a2 - 3 * a3 * a1;
				if (d >= 0.0)
				{
					d = std::sqrt(d);
					double x1 = (-a2 - d) / (3 * a3);
					double x2 = (-a2 + d) / (3 * a3);
					double y1 = a0 + (a1 + (a2 + a3 * x1) * x1) * x1;
					double y2 = a0 + (a1 + (a2 + a3 * x2) * x2) * x2;
					if (y1 < y2)
						x_min = x1;
					else
						x_min = x2;
				}
			}
			// f'(x) = a1 + 2 * a2 * x = 0
			else if (std::abs(a2) > _eps)
				x_min = -a1 / (2 * a2);
			if (x_min < 0.0 || x_min > 1.0)
				x_min = (limit_max + limit_max) / 2;
			else if (x_min < limit_min)
				x_min = limit_min;
			else if (x_min > limit_max)
				x_min = limit_max;
			// Apply scale to the result.
			return a + x_min * (b - a);
		}

		// Value and derivative of the function f(x + a * p).
		// f - function.
		// x - initial point.
		// p - search direction.
		// a - step value.
		// xa = x + a * p (result).
		// fa = f(x + a * p) (result).
		// ga = df(x + a * p) / da (result).
		void _f_info(
			const std::function<DVal<N> (DVal<N>*, uint32_t)>& f,
			const double* const x,
			const double* const p,
			const double a,
			double* const xa,
			double& fa,
			double& ga,
			double* const g)
		{
			_add_v_av(x, p, a, xa);
			fa = _f_info(f, xa, g);
			ga = _mull_v_v(g, p);
		}

		inline double _f_info(const std::function<DVal<N> (DVal<N>*, uint32_t)>& f, double* const x, double* const g)
		{
			for (uint32_t i = 0; i < N; ++i)
				_dval[i].set(x[i], i);
			const auto r = f(_dval, N);
			for (uint32_t i = 0; i < N; ++i)
				g[i] = r.d[i + 1];
			return r.d[0];
		}

		// Line search.
		// f - function.
		// x - initial point.
		// p - search direction.
		// y = f(x).
		// d = grad(f(x))^T * p.
		// xi = x + ai * p - result point.
		// fi = f(xi) - function value in result point.
		// ai - step value.
		// g = grad(f(xi)) (if the derivatives are analytical).
		template <typename fun>
		double _line_search(
				const fun& f,
				const double* const x,
				const double* const p,
				const double y,
				const double d,
				double* const xi,
				double& fi,
				double& ai,
				double* const g)
		{
			// 0 < c1 < c2 < 1
			// Strong Wolfe conditions.
			// f(x + a * p) <= f(x) + c1 * a * grad(f(x))^T * p
			// |grad(f(x + a * p))^T * p| <= c2 * |grad(f(x))^T * p|
			// Normal Wolfe condition.
			// f(x + a * p) <= f(x) + c1 * a * grad(f(x))^T * p
			// grad(f(x + a * p))^T * p >= c2 * grad(f(x))^T * p
			// Preparing constants.
			// d = grad(f(x))^T * p
			// k1 = c1 * d = c1 * grad(f(x))^T * p
			// k2 = c2 * |d| = c2 * |grad(f(x))^T * p|
			const double k1 = _c1 * d;
			const double k2 = _use_strong_wolfe ? _c2 * std::abs(d) : _c2 * d;
			// Search the range [a, b], a < b.
			double a = 0.0;
			// fa = f(a)
			double fa = y;
			// ga = grad(f(a))^T * p
			double ga = d;
			// Initial value.
			double b = 1.0;
			// Step tweak.
			if (_step_tweak > 0.0 && y > _min_f)
			{
				b = _step_tweak * (_min_f - y) / d;
				if (b > 1.0)
					b = 1.0;
			}
			// fb = f(b)
			double fb;
			// gb = grad(f(b))^T * p
			double gb;
			//
			uint32_t line_iter = 0;
			for (; line_iter < _line_iter_max; ++line_iter)
			{
				_f_info(f, x, p, b, xi, fb, gb, g);
				// 
				if (fb > fa || fb > y + b * k1)
					break;
				// Wolfe conditions is satisfied.
				if ((_use_strong_wolfe && std::abs(gb) < k2) || (!_use_strong_wolfe && gb > k2))
				{
					fi = fb;
					ai = b;
					return y - fi;
				}
				// Stop if ga < 0 and gb > 0.
				// There is the minimum in this range.
				if (gb > 0)
					break;
				// 
				a = b;
				fa = fb;
				ga = gb;
				// TODO. Extrapolation.
				// Simple way increase b.
				b *= _db;
			}
			//
			double gi;
			for (; line_iter < _line_iter_max; ++line_iter)
			{
				// TODO. Some thresholds to avoid extreme points.
				ai = _poly(a, b, fa, ga, fb, gb, 0.1, 0.9);
				_f_info(f, x, p, ai, xi, fi, gi, g);
				//
				if (fi > fa || fi > y + b * k1)
				{
					b = ai;
					fb = fi;
					gb = gi;
					continue;
				}
				// Wolfe conditions is satisfied.
				if ((_use_strong_wolfe && std::abs(gi) < k2) || (!_use_strong_wolfe && gi > k2))
					break;
				if (gi < 0)
				{
					a = ai;
					fa = fi;
					ga = gi;
				}
				else
				{
					b = ai;
					fb = fi;
					gb = gi;
				}
			}
			return y - fi;
		}

	public:
		BFGS()
		{
			_diag(_h);
		}

		~BFGS()
		{
		}

		// Stop criterion. Gradient Norm.
		// Defaul: 1e-7.
		void set_stop_grad_eps(double stop_grad_eps)
		{
			if (_stop_grad_eps > 0.0)
				_stop_grad_eps = stop_grad_eps;
		}

		// Stop criterion. The amount the feature changes between iterations.
		// Defaul: 1e-7.
		void set_stop_step_eps(double stop_step_eps)
		{
			if (stop_step_eps > 0.0)
				_stop_step_eps = stop_step_eps;
		}

		// Epsilon for calculating gradients.
		// It is also used for some other checks.
		// Defaul: 1e-8.
		void set_grad_eps(double grad_eps)
		{
			if (grad_eps > 0.0)
				_eps = grad_eps;
		}

		// Stop criterion. The maximum number of iterations of the algorithm.
		void set_max_iter(uint32_t max_iter)
		{
			if (max_iter > 0)
				_max_iter = max_iter;
		}

		// Wolfe criterion constants.
		// 0.0 < c1 < c2 < 1.0
		// Defaul: 0.01 and 0.9.
		void set_c1_c2(double c1, double c2)
		{
			if (c1 > 0.0 && c2 > c1 && c2 < 1.0)
			{
				_c1 = c1;
				_c2 = c2;
			}
		}

		// The maximum number of iterations of the line search.
		// Defaul: 100.
		void set_line_max_iter(uint32_t line_max_iter)
		{
			if (line_max_iter > 0)
				_line_iter_max = line_max_iter;
		}

		// Not used.
		// Use central difference or forward difference for the gradient.
		// Defaul: true.
		void set_central_diff(bool central_diff)
		{
		}

		// Not used.
		// Use central difference or forward difference for the line serach.
		// Maybe we don't want too much precision here.
		// You can try it to reduce the number of function calls.
		// Defaul: true.
		void set_line_central_diff(bool line_central_diff)
		{
		}

		// Estimated minimum of the function.
		// Used for stop criterion if fk - grad_eps < min_f
		// Also used for step_tweak option.
		// Defaul: inf (not use).
		void set_min_f(double min_f = std::numeric_limits<double>::infinity())
		{
			_min_f = min_f;
		}

		// Experimental setting of the first step of the linear search.
		// If set, then the first step of the linear search will be:
		// a = min(1, step_tweak * (f_min - f) / drad(f))
		// Defaul: 0.0 (not use if <= 0.0).
		void set_step_tweak(double step_tweak)
		{
			_step_tweak = step_tweak;
		}

		// Experimental setting.
		// When solving many typical problems, the value from the previous run
		// can be used as the initial approximation of the inverse Hessian.
		// Defaul: false.
		void set_reuse_hessian(bool reuse_hessian)
		{
			_reuse_hessian = reuse_hessian;
		}

		// Not used.
		// If we want to save some memory for a large problem.
		// In the usual case,
		//    the memory size is proportional to n * (n + 4).
		// In the case of storing only the upper triangular inverse Hessian matrix,
		//    the memory size is proportional to n * (n + 9) / 2.
		// Defaul: false.
		void set_memory_save(bool memory_save)
		{
		}

		// Use Strong Wolfe conditions.
		// Defaul: true.
		void set_strong_wolfe(bool use_strong_wolfe)
		{
			_use_strong_wolfe = use_strong_wolfe;
		}

		// Not used.
		// Specifies the amount of memory for automatic derivative
		// Defaul: 100.
		void set_dval_size(uint32_t dval_size)
		{
		}

		// Search for the function minimum.
		// f - minimized function
		//    automatic derivative: DVal<N> f(DVal<N>* x, uint32_t n)
		// x - initial point value and result.
		// n - dimension.
		double find_min(const std::function<DVal<N> (DVal<N>*, uint32_t)>& f, double* const x, const uint32_t n)
		{
			//
			if (n != N || x == nullptr)
				return std::numeric_limits<double>::infinity();
			//
			if (!_reuse_hessian)
				_diag(_h);
			// Step value.
			double ai;
			// Function value and derivatives (if the derivatives are analytical).
			// y = f(x)
			// g = grad(f(x))
			double y = _f_info(f, x, _g);
			// double y = f(x, _n);
			// Limit the number of iterations.
			uint32_t iter = 0;
			for (; iter < _max_iter; ++iter)
			{
				// Stop check.
				if (std::isfinite(_min_f) && y - _eps < _min_f)
					break;
				// Calculation of the gradient.
				// g = grad(f(x))
				// L2 norm of the gradient.
				// g_norm = |g|.
				double g_norm = _norm(_g);
				// Stop check.
				if (g_norm < _stop_grad_eps)
					break;
				// Update the inverse hessian if it's not the first iteration.
				if (iter > 0)
				{
					// The gradient increment.
					// dg = g - dg.
					_sub_v_v(_g, _dg, _dg);
					// p_dg = p^T * dg
					const double p_dg = _mull_v_v(_p, _dg);
					if (p_dg > _eps)
					{
						// Inverse Hessian update.
						// h = h + (s^T * dg + dg^T * h * dg) * (s * s^T) / (s^T * dg)^2 - (h * dg * s^T + s * dg^T * h) / (s^T * dg)
						// s = a * p
						// h = h + (a * p^T * dg + dg^T * h * dg) * (p * p^T) / (p^T * dg)^2 - (h * dg * p^T + p * dg^T * h) / (p^T * dg)
						// pdg = 1 / (p^T * dg)
						const double pdg = 1.0 / p_dg;
						// xi = hdg = h * dg = dg^T * h (h - symmetric matrix)
						_mull_m_v(_h, _dg, _xi);
						// The expression now looks like:
						// h = h + (a * pdg + pdg^2 * dg^T * hdg) * (p * p^T) - pdg * (hdg * p^T + p * hdg^T)
						// tmp = a * pdg + pdg^2 * dg^T * hdg
						const double tmp = pdg * (ai + pdg * _mull_v_v(_dg, _xi));
						// h = h + tmp * (p * p^T) - pdg * (hdg * p^T + p * hdg^T)
						for (uint32_t i = 0, m = 0; i < N; ++i)
						{
							// h - symmetric matrix
							for (uint32_t j = 0; j < i; ++j, ++m)
								_h[m] = _h[j * N + i];
							for (uint32_t j = i; j < N; ++j, ++m)
								_h[m] += tmp * _p[i] * _p[j] - pdg * (_xi[i] * _p[j] + _p[i] * _xi[j]);
						}
					}
				}
				// Search direction.
				// p = h * g
				_mull_m_v(_h, _g, _p);
				// p = -p
				for (uint32_t i = 0; i < N; ++i)
					_p[i] = -_p[i];
				// d = g^T * p (the derivative at the point a = 0.0).
				const double d = _mull_v_v(_g, _p);
				if (std::abs(d) < std::abs(y) * std::numeric_limits<double>::epsilon())
					break;
				// dg = g
				_copy_v(_g, _dg);
				// Line search.
				// If the derivatives are analytical, then its value will be stored in _g.
				double dy = _line_search(f, x, _p, y, d, _xi, y, ai, _g);
				// x = xi
				_copy_v(_xi, x);
				// Stop check.
				if (std::abs(dy) < _stop_step_eps)
					break;
			}
			// y = f(х)
			return y;
		}
	};
}