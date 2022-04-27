// Copyright © 2022 Alexander Abramenkov. All rights reserved.
// Licensed under the Apache License, Version 2.0.
// https://github.com/IOdissey/bfgs

#pragma once

#include <cmath>
#include <cstring>
#include <exception>

#include "memory.h"


// Class for automatic differentiation with static dimensions.
template <uint32_t N>
struct DVal
{
	double d[N + 1];

	DVal()
	{
	}

	DVal(const DVal<N>& x, const bool& copy = true)
	{
		if (copy)
			std::memcpy(d, x.d, n() * sizeof(double));
	}

	DVal(const double& v, const uint32_t& j)
	{
		set(v, j);
	}

	~DVal()
	{
	}

	constexpr uint32_t n() const
	{
		return N + 1;
	}

	inline void set(const double& v)
	{
		d[0] = v;
	}

	inline void set_j(const uint32_t& j)
	{
		for (uint32_t i = 1; i < n(); ++i)
			d[i] = 0.0;
		d[j + 1] = 1.0;
	}

	inline void set(const double& v, const uint32_t& j)
	{
		set(v);
		set_j(j);
	}

	// For compatibility with DVal<0>.
	inline void set(const double& v, const uint32_t& j, Memory* mem)
	{
		set(v, j);
	}

	inline DVal<N>& operator=(const DVal<N>& x)
	{
		std::memcpy(d, x.d, n() * sizeof(double));
		return *this;
	}

	inline DVal<N>& operator+=(const DVal<N>& x)
	{
		for (uint32_t i = 0; i < n(); ++i)
			d[i] += x.d[i];
		return *this;
	}

	inline DVal<N>& operator+=(const double& c)
	{
		d[0] += c;
		return *this;
	}

	inline DVal<N>& operator-=(const DVal<N>& x)
	{
		for (uint32_t i = 0; i < n(); ++i)
			d[i] -= x.d[i];
		return *this;
	}

	inline DVal<N>& operator-=(const double& c)
	{
		d[0] -= c;
		return *this;
	}

	inline DVal<N>& operator*=(const DVal<N>& x)
	{
		for (uint32_t i = 1; i < n(); ++i)
			d[i] = d[i] * x.d[0] + d[0] * x.d[i];
		d[0] *= x.d[0];
		return *this;
	}

	inline DVal<N>& operator*=(const double& c)
	{
		for (uint32_t i = 0; i < n(); ++i)
			d[i] *= c;
		return *this;
	}

	inline DVal<N>& operator/=(const DVal<N>& x)
	{
		const double v_inv = 1.0 / x.d[0];
		d[0] *= v_inv;
		for (uint32_t i = 1; i < n(); ++i)
			d[i] = (d[i] - d[0] * x.d[i]) * v_inv;
		return *this;
	}

	inline DVal<N>& operator/=(const double& c)
	{
		const double c_inv = 1.0 / c;
		for (uint32_t i = 0; i < n(); ++i)
			d[i] *= c_inv;
		return *this;
	}
};

// Class for automatic differentiation with dinamic dimensions.
template<>
struct DVal<0>
{
	Memory* m;
	double* d;

	DVal():
		m(nullptr)
	{
	}

	DVal(const DVal<0>& x, const bool& copy = true):
		m(x.m)
	{
		if (m == nullptr)
			throw std::runtime_error("DVal can't set Memory.");
		d = m->create();
		if (copy)
			std::memcpy(d, x.d, n() * sizeof(double));
	}

	DVal(const double& v, const uint32_t& j, Memory* mem):
		m(mem)
	{
		d = m->create();
		set(v, j);
	}

	~DVal()
	{
		if (m)
			m->free(d);
	}

	inline const uint32_t& n() const
	{
		return m->n();
	}

	inline void set(Memory* mem)
	{
		if (m != nullptr || mem == nullptr)
			throw std::runtime_error("DVal can't set Memory.");
		m = mem;
		d = m->create();
	}

	inline void set(const double& v)
	{
		d[0] = v;
	}

	inline void set_j(const uint32_t& j)
	{
		const uint32_t nn = n();
		for (uint32_t i = 1; i < nn; ++i)
			d[i] = 0.0;
		d[j + 1] = 1.0;
	}

	inline void set(const double& v, const uint32_t& j)
	{
		set(v);
		set_j(j);
	}

	inline void set(const double& v, const uint32_t& j, Memory* mem)
	{
		set(mem);
		set(v, j);
	}

	inline DVal<0>& operator=(const DVal<0>& x)
	{
		set(x.m);
		std::memcpy(d, x.d, n() * sizeof(double));
		return *this;
	}

	inline DVal<0>& operator+=(const DVal<0>& x)
	{
		const uint32_t nn = n();
		for (uint32_t i = 0; i < nn; ++i)
			d[i] += x.d[i];
		return *this;
	}

	inline DVal<0>& operator+=(const double& c)
	{
		d[0] += c;
		return *this;
	}

	inline DVal<0>& operator-=(const DVal<0>& x)
	{
		const uint32_t nn = n();
		for (uint32_t i = 0; i < nn; ++i)
			d[i] -= x.d[i];
		return *this;
	}

	inline DVal<0>& operator-=(const double& c)
	{
		d[0] -= c;
		return *this;
	}

	inline DVal<0>& operator*=(const DVal<0>& x)
	{
		const uint32_t nn = n();
		for (uint32_t i = 1; i < nn; ++i)
			d[i] = d[i] * x.d[0] + d[0] * x.d[i];
		d[0] *= x.d[0];
		return *this;
	}

	inline DVal<0>& operator*=(const double& c)
	{
		const uint32_t nn = n();
		for (uint32_t i = 0; i < nn; ++i)
			d[i] *= c;
		return *this;
	}

	inline DVal<0>& operator/=(const DVal<0>& x)
	{
		const double v_inv = 1.0 / x.d[0];
		d[0] *= v_inv;
		const uint32_t nn = n();
		for (uint32_t i = 1; i < nn; ++i)
			d[i] = (d[i] - d[0] * x.d[i]) * v_inv;
		return *this;
	}

	inline DVal<0>& operator/=(const double& c)
	{
		const double c_inv = 1.0 / c;
		const uint32_t nn = n();
		for (uint32_t i = 0; i < nn; ++i)
			d[i] *= c_inv;
		return *this;
	}
};

template <uint32_t N>
inline DVal<N> operator-(const DVal<N>& x)
{
	DVal<N> dval(x, false);
	const uint32_t nn = x.n();
	for (uint32_t i = 0; i < nn; ++i)
		dval.d[i] = -x.d[i];
	return dval;
}

template <uint32_t N>
inline DVal<N> operator+(const DVal<N>& x, const DVal<N>& y)
{
	DVal<N> dval(x, false);
	const uint32_t nn = x.n();
	for (uint32_t i = 0; i < nn; ++i)
		dval.d[i] = x.d[i] + y.d[i];
	return dval;
}

template <uint32_t N>
inline DVal<N> operator+(const DVal<N>& x, const double& c)
{
	DVal<N> dval(x, true);
	dval.d[0] += c;
	return dval;
}

template <uint32_t N>
inline DVal<N> operator+(const double& c, const DVal<N>& x)
{
	DVal<N> dval(x, true);
	dval.d[0] += c;
	return dval;
}

template <uint32_t N>
inline DVal<N> operator-(const DVal<N>& x, const DVal<N>& y)
{
	DVal<N> dval(x, false);
	const uint32_t nn = x.n();
	for (uint32_t i = 0; i < nn; ++i)
		dval.d[i] = x.d[i] - y.d[i];
	return dval;
}

template <uint32_t N>
inline DVal<N> operator-(const DVal<N>& x, const double& c)
{ 
	DVal<N> dval(x, true);
	dval.d[0] -= c;
	return dval;
}

template <uint32_t N>
inline DVal<N> operator-(const double& c, const DVal<N>& x)
{
	DVal<N> dval(x, false);
	dval.d[0] = c - x.d[0];
	const uint32_t nn = x.n();
	for (uint32_t i = 1; i < nn; ++i)
		dval.d[i] = -x.d[i];
	return dval;
}

template <uint32_t N>
inline DVal<N> operator*(const DVal<N>& x, const DVal<N>& y)
{
	DVal<N> dval(x, false);
	dval.d[0] = x.d[0] * y.d[0];
	const uint32_t nn = x.n();
	for (uint32_t i = 1; i < nn; ++i)
		dval.d[i] = x.d[i] * y.d[0] + x.d[0] * y.d[i];
	return dval;
}

template <uint32_t N>
inline DVal<N> operator*(const DVal<N>& x, const double& c)
{ 
	DVal<N> dval(x, false);
	const uint32_t nn = x.n();
	for (uint32_t i = 0; i < nn; ++i)
		dval.d[i] = x.d[i] * c;
	return dval;
}

template <uint32_t N>
inline DVal<N> operator*(const double& c, const DVal<N>& x)
{
	DVal<N> dval(x, false);
	const uint32_t nn = x.n();
	for (uint32_t i = 0; i < nn; ++i)
		dval.d[i] = x.d[i] * c;
	return dval;
}

template <uint32_t N>
inline DVal<N> operator/(const DVal<N>& x, const DVal<N>& y)
{
	DVal<N> dval(x, false);
	const double v_inv = 1.0 / y.d[0];
	dval.d[0] = x.d[0] * v_inv;
	const uint32_t nn = x.n();
	for (uint32_t i = 1; i < nn; ++i)
		dval.d[i] = (x.d[i] - dval.d[0] * y.d[i]) * v_inv;
	return dval;
}

template <uint32_t N>
inline DVal<N> operator/(const DVal<N>& x, const double& c)
{
	DVal<N> dval(x, false);
	const double c_inv = 1.0 / c;
	const uint32_t nn = x.n();
	for (uint32_t i = 0; i < nn; ++i)
		dval.d[i] = x.d[i] * c_inv;
	return dval;
}

template <uint32_t N>
inline DVal<N> operator/(const double& c, const DVal<N>& x)
{
	DVal<N> dval(x, false);
	dval.d[0] = c / x.d[0];
	const double v_inv = -dval.d[0] / x.d[0];
	const uint32_t nn = x.n();
	for (uint32_t i = 1; i < nn; ++i)
		dval.d[i] = x.d[i] * v_inv;
	return dval;
}

namespace std
{
	template <uint32_t N>
	inline DVal<N> abs(const DVal<N>& x)
	{
		if (x.d[0] < 0.0)
		{
			DVal<N> dval(x, false);
			const uint32_t nn = x.n();
			for (uint32_t i = 0; i < nn; ++i)
				dval.d[i] = -x.d[i];
			return dval;
		}
		else
		{
			DVal<N> dval(x, true);
			return dval;
		}
	}

	template <uint32_t N>
	inline DVal<N> sin(const DVal<N>& x)
	{
		DVal<N> dval(x, false);
		dval.d[0] = std::sin(x.d[0]);
		const double c = std::cos(x.d[0]);
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline DVal<N> cos(const DVal<N>& x)
	{
		DVal<N> dval(x, false);
		dval.d[0] = std::cos(x.d[0]);
		const double c = -std::sin(x.d[0]);
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline DVal<N> tan(const DVal<N>& x)
	{
		DVal<N> dval(x, false);
		dval.d[0] = std::tan(x.d[0]);
		const double c = 1.0 + dval.d[0] * dval.d[0];
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline DVal<N> asin(const DVal<N>& x)
	{
		DVal<N> dval(x, false);
		dval.d[0] = std::asin(x.d[0]);
		const double c = 1.0 / std::sqrt(1.0 - x.d[0] * x.d[0]);
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline DVal<N> acos(const DVal<N>& x)
	{
		DVal<N> dval(x, false);
		dval.d[0] = std::acos(x.d[0]);
		const double c = -1.0 / std::sqrt(1.0 - x.d[0] * x.d[0]);
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline DVal<N> atan(const DVal<N>& x)
	{
		DVal<N> dval(x, false);
		dval.d[0] = std::atan(x.d[0]);
		const double c = 1.0 / (1.0 + x.d[0] * x.d[0]);
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline DVal<N> log(const DVal<N>& x)
	{
		DVal<N> dval(x, false);
		dval.d[0] = std::log(x.d[0]);
		const double c = 1.0 / x.d[0];
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline DVal<N> log10(const DVal<N>& x)
	{
		DVal<N> dval(x, false);
		dval.d[0] = std::log10(x.d[0]);
		constexpr double k = 1.0 / std::log(10.0);
		const double c = k / x.d[0];
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline DVal<N> log2(const DVal<N>& x)
	{
		DVal<N> dval(x, false);
		dval.d[0] = std::log2(x.d[0]);
		constexpr double k = 1.0 / std::log(2.0);
		const double c = k / x.d[0];
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline DVal<N> exp(const DVal<N>& x)
	{
		DVal<N> dval(x, false);
		const double c = std::exp(x.d[0]);
		dval.d[0] = c;
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	// x.d[0] > 0
	template <uint32_t N>
	inline DVal<N> pow(const DVal<N>& x, const double& n)
	{
		DVal<N> dval(x, false);
		dval.d[0] = std::pow(x.d[0], n);
		const double c = n * dval.d[0] / x.d[0];
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	// a > 0.0
	template <uint32_t N>
	inline DVal<N> pow(const double& a, const DVal<N>& x)
	{
		DVal<N> dval(x, false);
		dval.d[0] = std::pow(a, x.d[0]);
		const double c = dval.d[0] * std::log(a);
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	// x.d[0] > 0.0
	template <uint32_t N>
	inline DVal<N> sqrt(const DVal<N>& x)
	{
		DVal<N> dval(x, false);
		dval.d[0] = std::sqrt(x.d[0]);
		const double c = 1.0 / (2.0 * dval.d[0]);
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	// x.d[0] != 0.0
	template <uint32_t N>
	inline DVal<N> cbrt(const DVal<N>& x)
	{
		DVal<N> dval(x, false);
		dval.d[0] = std::cbrt(x.d[0]);
		const double c = 1.0 / (3.0 * dval.d[0] * dval.d[0]);
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline ostream& operator<<(std::ostream& s, const DVal<N>& x)
	{
		s << x.d[0] << " (";
		const uint32_t nn = x.n();
		for (uint32_t i = 1; i < nn; ++i)
		{
			if (i > 1)
				s << ", ";
			s << x.d[i];
		}
		s << ")";
		return s;
	}
}