// Copyright © 2022 Alexander Abramenkov. All rights reserved.
// Licensed under the Apache License, Version 2.0.
// https://github.com/IOdissey/bfgs

#pragma once

#include <cmath>
#include <cstring>

#include "memory.h"


namespace bfgs::fix
{
	// Class for automatic differentiation with static dimensions.
	template <uint32_t N>
	struct DVal
	{
		double d[N + 1];

		DVal()
		{
		}

		DVal(const DVal<N>& x)
		{
			std::memcpy(d, x.d, (N + 1) * sizeof(double));
		}

		DVal(const double& val, const uint32_t& j)
		{
			set(val, j);
		}

		~DVal()
		{
		}

		void set(const double& val, const uint32_t& j)
		{
			d[0] = val;
			for (uint32_t i = 1; i < (N + 1); ++i)
				d[i] = 0.0;
			d[j + 1] = 1.0;
		}

		inline DVal<N>& operator=(const DVal<N>& x)
		{
			std::memcpy(d, x.d, (N + 1) * sizeof(double));
			return *this;
		}

		inline DVal<N>& operator+=(const DVal<N>& x)
		{
			for (uint32_t i = 0; i < (N + 1); ++i)
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
			for (uint32_t i = 0; i < (N + 1); ++i)
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
			for (uint32_t i = 1; i < (N + 1); ++i)
				d[i] = d[i] * x.d[0] + d[0] * x.d[i];
			d[0] *= x.d[0];
			return *this;
		}

		inline DVal<N>& operator*=(const double& c)
		{
			for (uint32_t i = 0; i < (N + 1); ++i)
				d[i] *= c;
			return *this;
		}

		inline DVal<N>& operator/=(const DVal<N>& x)
		{
			const double v_inv = 1.0 / x.d[0];
			d[0] *= v_inv;
			for (uint32_t i = 1; i < (N + 1); ++i)
				d[i] = (d[i] - d[0] * x.d[i]) * v_inv;
			return *this;
		}

		inline DVal<N>& operator/=(const double& c)
		{
			const double c_inv = 1.0 / c;
			for (uint32_t i = 0; i < (N + 1); ++i)
				d[i] *= c_inv;
			return *this;
		}
	};
}

template <uint32_t N>
inline bfgs::fix::DVal<N> operator-(const bfgs::fix::DVal<N>& x)
{
	bfgs::fix::DVal<N> dval;
	for (uint32_t i = 0; i < (N + 1); ++i)
		dval.d[i] = -x.d[i];
	return dval;
}

template <uint32_t N>
inline bfgs::fix::DVal<N> operator+(const bfgs::fix::DVal<N>& x, const bfgs::fix::DVal<N>& y)
{
	bfgs::fix::DVal<N> dval;
	for (uint32_t i = 0; i < (N + 1); ++i)
		dval.d[i] = x.d[i] + y.d[i];
	return dval;
}

template <uint32_t N>
inline bfgs::fix::DVal<N> operator+(const bfgs::fix::DVal<N>& x, const double& c)
{
	bfgs::fix::DVal<N> dval(x);
	dval.d[0] += c;
	return dval;
}

template <uint32_t N>
inline bfgs::fix::DVal<N> operator+(const double& c, const bfgs::fix::DVal<N>& x)
{
	bfgs::fix::DVal<N> dval(x);
	dval.d[0] += c;
	return dval;
}

template <uint32_t N>
inline bfgs::fix::DVal<N> operator-(const bfgs::fix::DVal<N>& x, const bfgs::fix::DVal<N>& y)
{
	bfgs::fix::DVal<N> dval;
	for (uint32_t i = 0; i < (N + 1); ++i)
		dval.d[i] = x.d[i] - y.d[i];
	return dval;
}

template <uint32_t N>
inline bfgs::fix::DVal<N> operator-(const bfgs::fix::DVal<N>& x, const double& c)
{ 
	bfgs::fix::DVal<N> dval(x);
	dval.d[0] -= c;
	return dval;
}

template <uint32_t N>
inline bfgs::fix::DVal<N> operator-(const double& c, const bfgs::fix::DVal<N>& x)
{
	bfgs::fix::DVal<N> dval;
	dval.d[0] = c - x.d[0];
	for (uint32_t i = 1; i < (N + 1); ++i)
		dval.d[i] = -x.d[i];
	return dval;
}

template <uint32_t N>
inline bfgs::fix::DVal<N> operator*(const bfgs::fix::DVal<N>& x, const bfgs::fix::DVal<N>& y)
{
	bfgs::fix::DVal<N> dval;
	dval.d[0] = x.d[0] * y.d[0];
	for (uint32_t i = 1; i < (N + 1); ++i)
		dval.d[i] = x.d[i] * y.d[0] + x.d[0] * y.d[i];
	return dval;
}

template <uint32_t N>
inline bfgs::fix::DVal<N> operator*(const bfgs::fix::DVal<N>& x, const double& c)
{ 
	bfgs::fix::DVal<N> dval;
	for (uint32_t i = 0; i < (N + 1); ++i)
		dval.d[i] = x.d[i] * c;
	return dval;
}

template <uint32_t N>
inline bfgs::fix::DVal<N> operator*(const double& c, const bfgs::fix::DVal<N>& x)
{
	bfgs::fix::DVal<N> dval;
	for (uint32_t i = 0; i < (N + 1); ++i)
		dval.d[i] = x.d[i] * c;
	return dval;
}

template <uint32_t N>
inline bfgs::fix::DVal<N> operator/(const bfgs::fix::DVal<N>& x, const bfgs::fix::DVal<N>& y)
{
	bfgs::fix::DVal<N> dval;
	const double v_inv = 1.0 / y.d[0];
	dval.d[0] = x.d[0] * v_inv;
	for (uint32_t i = 1; i < (N + 1); ++i)
		dval.d[i] = (x.d[i] - dval.d[0] * y.d[i]) * v_inv;
	return dval;
}

template <uint32_t N>
inline bfgs::fix::DVal<N> operator/(const bfgs::fix::DVal<N>& x, const double& c)
{
	bfgs::fix::DVal<N> dval;
	const double c_inv = 1.0 / c;
	for (uint32_t i = 0; i < (N + 1); ++i)
		dval.d[i] = x.d[i] * c_inv;
	return dval;
}

template <uint32_t N>
inline bfgs::fix::DVal<N> operator/(const double& c, const bfgs::fix::DVal<N>& x)
{
	bfgs::fix::DVal<N> dval;
	dval.d[0] = c / x.d[0];
	const double v_inv = -dval.d[0] / x.d[0];
	for (uint32_t i = 1; i < (N + 1); ++i)
		dval.d[i] = x.d[i] * v_inv;
	return dval;
}

namespace std
{
	template <uint32_t N>
	inline bfgs::fix::DVal<N> abs(const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		if (x.d[0] < 0.0)
		{
			for (uint32_t i = 0; i < (N + 1); ++i)
				dval.d[i] = -x.d[i];
		}
		else
		{
			for (uint32_t i = 0; i < (N + 1); ++i)
				dval.d[i] = x.d[i];
		}
		return dval;
	}

	template <uint32_t N>
	inline bfgs::fix::DVal<N> sin(const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		dval.d[0] = std::sin(x.d[0]);
		const double c = std::cos(x.d[0]);
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline bfgs::fix::DVal<N> cos(const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		dval.d[0] = std::cos(x.d[0]);
		const double c = -std::sin(x.d[0]);
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline bfgs::fix::DVal<N> tan(const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		dval.d[0] = std::tan(x.d[0]);
		const double c = 1.0 + dval.d[0] * dval.d[0];
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline bfgs::fix::DVal<N> asin(const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		dval.d[0] = std::asin(x.d[0]);
		const double c = 1.0 / std::sqrt(1.0 - x.d[0] * x.d[0]);
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline bfgs::fix::DVal<N> acos(const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		dval.d[0] = std::acos(x.d[0]);
		const double c = -1.0 / std::sqrt(1.0 - x.d[0] * x.d[0]);
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline bfgs::fix::DVal<N> atan(const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		dval.d[0] = std::atan(x.d[0]);
		const double c = 1.0 / (1.0 + x.d[0] * x.d[0]);
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline bfgs::fix::DVal<N> log(const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		dval.d[0] = std::log(x.d[0]);
		const double c = 1.0 / x.d[0];
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline bfgs::fix::DVal<N> log10(const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		dval.d[0] = std::log10(x.d[0]);
		constexpr double k = 1.0 / std::log(10.0);
		const double c = k / x.d[0];
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline bfgs::fix::DVal<N> log2(const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		dval.d[0] = std::log2(x.d[0]);
		constexpr double k = 1.0 / std::log(2.0);
		const double c = k / x.d[0];
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline bfgs::fix::DVal<N> exp(const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		const double c = std::exp(x.d[0]);
		dval.d[0] = c;
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	// x.d[0] > 0
	template <uint32_t N>
	inline bfgs::fix::DVal<N> pow(const bfgs::fix::DVal<N>& x, const double& n)
	{
		bfgs::fix::DVal<N> dval;
		dval.d[0] = std::pow(x.d[0], n);
		const double c = n * dval.d[0] / x.d[0];
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	// a > 0.0
	template <uint32_t N>
	inline bfgs::fix::DVal<N> pow(const double& a, const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		dval.d[0] = std::pow(a, x.d[0]);
		const double c = dval.d[0] * std::log(a);
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	// x.d[0] > 0.0
	template <uint32_t N>
	inline bfgs::fix::DVal<N> sqrt(const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		dval.d[0] = std::sqrt(x.d[0]);
		const double c = 1.0 / (2.0 * dval.d[0]);
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	// x.d[0] != 0.0
	template <uint32_t N>
	inline bfgs::fix::DVal<N> cbrt(const bfgs::fix::DVal<N>& x)
	{
		bfgs::fix::DVal<N> dval;
		dval.d[0] = std::cbrt(x.d[0]);
		const double c = 1.0 / (3.0 * dval.d[0] * dval.d[0]);
		for (uint32_t i = 1; i < (N + 1); ++i)
			dval.d[i] = x.d[i] * c;
		return dval;
	}

	template <uint32_t N>
	inline ostream& operator<<(std::ostream& s, const bfgs::fix::DVal<N>& dval)
	{
		s << dval.d[0] << " (";
		for (uint32_t i = 1; i < (N + 1); ++i)
		{
			if (i > 1)
				s << ", ";
			s << dval.d[i];
		}
		s << ")";
		return s;
	}
}