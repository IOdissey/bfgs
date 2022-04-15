// Copyright © 2022 Alexander Abramenkov. All rights reserved.
// Licensed under the Apache License, Version 2.0.
// https://github.com/IOdissey/bfgs

#pragma once

#include <cmath>
#include <cstring>

#include "memory.h"


namespace bfgs::var
{
	// Class for automatic differentiation.
	struct DVal
	{
		Memory* m = nullptr;
		double* d;

		DVal()
		{
		}

		DVal(Memory* mem)
		{
			init(mem);
		}

		DVal(const DVal& x)
		{
			init(x.m);
			std::memcpy(d, x.d, n() * sizeof(double));
		}

		DVal(const double& val, const uint32_t& j, Memory* mem)
		{
			init(mem);
			set(val, j);
		}

		~DVal()
		{
			m->free(d);
		}

		// DVal& operator=(const DVal& x)
		// {
		// 	set(x.m);
		// 	std::memcpy(d, x.d, n() * sizeof(double));
		// 	return *this;
		// }

		inline const uint32_t& n() const
		{
			return m->n();
		}

		inline void init(Memory* mem)
		{
			m = mem;
			d = m->create();
		}

		inline void set(Memory* mem)
		{
			if (m == nullptr)
				init(mem);
		}

		void set(const double& val, const uint32_t& j)
		{
			d[0] = val;
			for (uint32_t i = 1; i < n(); ++i)
				d[i] = 0.0;
			d[j + 1] = 1.0;
		}

		void set(const double& val, const uint32_t& j, Memory* mem)
		{
			set(mem);
			set(val, j);
		}

		inline DVal& operator+=(const DVal& x)
		{
			for (uint32_t i = 0; i < n(); ++i)
				d[i] += x.d[i];
			return *this;
		}

		inline DVal& operator+=(const double& c)
		{
			d[0] += c;
			return *this;
		}

		inline DVal& operator-=(const DVal& x)
		{
			for (uint32_t i = 0; i < n(); ++i)
				d[i] -= x.d[i];
			return *this;
		}

		inline DVal& operator-=(const double& c)
		{
			d[0] -= c;
			return *this;
		}

		inline DVal& operator*=(const DVal& x)
		{
			for (uint32_t i = 1; i < n(); ++i)
				d[i] = d[i] * x.d[0] + d[0] * x.d[i];
			d[0] *= x.d[0];
			return *this;
		}

		inline DVal& operator*=(const double& c)
		{
			for (uint32_t i = 0; i < n(); ++i)
				d[i] *= c;
			return *this;
		}

		inline DVal& operator/=(const DVal& x)
		{
			const double v_inv = 1.0 / x.d[0];
			d[0] *= v_inv;
			for (uint32_t i = 1; i < n(); ++i)
				d[i] = (d[i] - d[0] * x.d[i]) * v_inv;
			return *this;
		}

		inline DVal& operator/=(const double& c)
		{
			const double c_inv = 1.0 / c;
			for (uint32_t i = 0; i < n(); ++i)
				d[i] *= c_inv;
			return *this;
		}
	};
}

inline bfgs::var::DVal operator-(const bfgs::var::DVal& x)
{
	bfgs::var::DVal dval(x.m);
	for (uint32_t i = 0; i < dval.n(); ++i)
		dval.d[i] = -x.d[i];
	return dval;
}

inline bfgs::var::DVal operator+(const bfgs::var::DVal& x, const bfgs::var::DVal& y)
{
	bfgs::var::DVal dval(x.m);
	for (uint32_t i = 0; i < dval.n(); ++i)
		dval.d[i] = x.d[i] + y.d[i];
	return dval;
}

inline bfgs::var::DVal operator+(const bfgs::var::DVal& x, const double& c)
{
	bfgs::var::DVal dval(x);
	dval.d[0] += c;
	return dval;
}

inline bfgs::var::DVal operator+(const double& c, const bfgs::var::DVal& x)
{
	bfgs::var::DVal dval(x);
	dval.d[0] += c;
	return dval;
}

inline bfgs::var::DVal operator-(const bfgs::var::DVal& x, const bfgs::var::DVal& y)
{
	bfgs::var::DVal dval(x.m);
	for (uint32_t i = 0; i < dval.n(); ++i)
		dval.d[i] = x.d[i] - y.d[i];
	return dval;
}

inline bfgs::var::DVal operator-(const bfgs::var::DVal& x, const double& c)
{ 
	bfgs::var::DVal dval(x);
	dval.d[0] -= c;
	return dval;
}

inline bfgs::var::DVal operator-(const double& c, const bfgs::var::DVal& x)
{
	bfgs::var::DVal dval(x.m);
	dval.d[0] = c - x.d[0];
	for (uint32_t i = 1; i < dval.n(); ++i)
		dval.d[i] = -x.d[i];
	return dval;
}

inline bfgs::var::DVal operator*(const bfgs::var::DVal& x, const bfgs::var::DVal& y)
{
	bfgs::var::DVal dval(x.m);
	dval.d[0] = x.d[0] * y.d[0];
	for (uint32_t i = 1; i < dval.n(); ++i)
		dval.d[i] = x.d[i] * y.d[0] + x.d[0] * y.d[i];
	return dval;
}

inline bfgs::var::DVal operator*(const bfgs::var::DVal& x, const double& c)
{ 
	bfgs::var::DVal dval(x.m);
	for (uint32_t i = 0; i < dval.n(); ++i)
		dval.d[i] = x.d[i] * c;
	return dval;
}

inline bfgs::var::DVal operator*(const double& c, const bfgs::var::DVal& x)
{
	bfgs::var::DVal dval(x.m);
	for (uint32_t i = 0; i < dval.n(); ++i)
		dval.d[i] = x.d[i] * c;
	return dval;
}

inline bfgs::var::DVal operator/(const bfgs::var::DVal& x, const bfgs::var::DVal& y)
{
	bfgs::var::DVal dval(x.m);
	const double v_inv = 1.0 / y.d[0];
	dval.d[0] = x.d[0] * v_inv;
	for (uint32_t i = 1; i < dval.n(); ++i)
		dval.d[i] = (x.d[i] - dval.d[0] * y.d[i]) * v_inv;
	return dval;
}

inline bfgs::var::DVal operator/(const bfgs::var::DVal& x, const double& c)
{
	bfgs::var::DVal dval(x.m);
	const double c_inv = 1.0 / c;
	for (uint32_t i = 0; i < dval.n(); ++i)
		dval.d[i] = x.d[i] * c_inv;
	return dval;
}

inline bfgs::var::DVal operator/(const double& c, const bfgs::var::DVal& x)
{
	bfgs::var::DVal dval(x.m);
	dval.d[0] = c / x.d[0];
	const double v_inv = -dval.d[0] / x.d[0];
	for (uint32_t i = 1; i < dval.n(); ++i)
		dval.d[i] = x.d[i] * v_inv;
	return dval;
}

namespace std
{
	inline bfgs::var::DVal abs(const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		if (x.d[0] < 0.0)
		{
			for (uint32_t i = 0; i < dval.n(); ++i)
				dval.d[i] = -x.d[i];
		}
		else
		{
			for (uint32_t i = 0; i < dval.n(); ++i)
				dval.d[i] = x.d[i];
		}
		return dval;
	}

	inline bfgs::var::DVal sin(const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		dval.d[0] = std::sin(x.d[0]);
		const double c = std::cos(x.d[0]);
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	inline bfgs::var::DVal cos(const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		dval.d[0] = std::cos(x.d[0]);
		const double c = -std::sin(x.d[0]);
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	inline bfgs::var::DVal tan(const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		dval.d[0] = std::tan(x.d[0]);
		const double c = 1.0 + dval.d[0] * dval.d[0];
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	inline bfgs::var::DVal asin(const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		dval.d[0] = std::asin(x.d[0]);
		const double c = 1.0 / std::sqrt(1.0 - x.d[0] * x.d[0]);
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	inline bfgs::var::DVal acos(const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		dval.d[0] = std::acos(x.d[0]);
		const double c = -1.0 / std::sqrt(1.0 - x.d[0] * x.d[0]);
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	inline bfgs::var::DVal atan(const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		dval.d[0] = std::atan(x.d[0]);
		const double c = 1.0 / (1.0 + x.d[0] * x.d[0]);
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	inline bfgs::var::DVal log(const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		dval.d[0] = std::log(x.d[0]);
		const double c = 1.0 / x.d[0];
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	inline bfgs::var::DVal log10(const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		dval.d[0] = std::log10(x.d[0]);
		constexpr double k = 1.0 / std::log(10.0);
		const double c = k / x.d[0];
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	inline bfgs::var::DVal log2(const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		dval.d[0] = std::log2(x.d[0]);
		constexpr double k = 1.0 / std::log(2.0);
		const double c = k / x.d[0];
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	inline bfgs::var::DVal exp(const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		const double c = std::exp(x.d[0]);
		dval.d[0] = c;
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	// x.d[0] > 0
	inline bfgs::var::DVal pow(const bfgs::var::DVal& x, const double& n)
	{
		bfgs::var::DVal dval(x.m);
		dval.d[0] = std::pow(x.d[0], n);
		const double c = n * dval.d[0] / x.d[0];
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	// a > 0.0
	inline bfgs::var::DVal pow(const double& a, const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		dval.d[0] = std::pow(a, x.d[0]);
		const double c = dval.d[0] * std::log(a);
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	// x.d[0] > 0.0
	inline bfgs::var::DVal sqrt(const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		dval.d[0] = std::sqrt(x.d[0]);
		const double c = 1.0 / (2.0 * dval.d[0]);
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	// x.d[0] != 0.0
	inline bfgs::var::DVal cbrt(const bfgs::var::DVal& x)
	{
		bfgs::var::DVal dval(x.m);
		dval.d[0] = std::cbrt(x.d[0]);
		const double c = 1.0 / (3.0 * dval.d[0] * dval.d[0]);
		for (uint32_t i = 1; i < dval.n(); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	inline ostream& operator<<(std::ostream& s, const bfgs::var::DVal& dval)
	{
		s << dval.d[0] << " (";
		for (uint32_t i = 1; i < dval.n(); ++i)
		{
			if (i > 1)
				s << ", ";
			s << dval.d[i];
		}
		s << ")";
		return s;
	}
}