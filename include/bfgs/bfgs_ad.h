// Copyright © 2022 Alexander Abramenkov. All rights reserved.
// Licensed under the Apache License, Version 2.0.
// https://github.com/IOdissey/bfgs

#pragma once

#include "bfgs.h"
#include "dval.h"
#include "memory.h"


// Version with automatic derivatives.
class BFGS_AD : public BFGS
{
private:
	uint32_t _dval_size = 100;      // Specifies the amount of memory for automatic derivative with dynamic dimension.

public:
	// Specifies the amount of memory for automatic derivative with dynamic dimension.
	// Defaul: 100.
	void set_dval_size(uint32_t dval_size)
	{
		_dval_size = dval_size;
	}

	template <uint32_t N>
	double find_min_auto(const std::function<DVal<N> (DVal<N>*, uint32_t)>& f, double* const x, const uint32_t n = N)
	{
		if (n != N)
			return std::numeric_limits<double>::infinity();
		_line_force_num = false;
		DVal<N> dval[N];
		for (uint32_t i = 0; i < N; ++i)
			dval[i].set(x[i], i);
		DVal<N>* ptr = dval;
		auto g = [&f, ptr](const double* const x, double* const g, const uint32_t n) -> double
		{
			for (uint32_t i = 0; i < n; ++i)
				ptr[i].set(x[i]);
			const DVal<N> r = f(ptr, n);
			for (uint32_t i = 0; i < n; ++i)
				g[i] = r.d[i + 1];
			return r.d[0];
		};
		return find_min(g, x, n);
	}

	double find_min_auto(const std::function<DVal<0> (DVal<0>*, uint32_t)>& f, double* const x, const uint32_t n)
	{
		if (n == 0)
			return std::numeric_limits<double>::infinity();
		_line_force_num = false;
		Memory mem(n, _dval_size);
		auto g = [&mem, &f](const double* const x, double* const g, uint32_t n) -> double
		{
			DVal<0>* dval = new DVal<0>[n];
			for (uint32_t i = 0; i < n; ++i)
				dval[i].set(x[i], i, &mem);
			DVal<0> r = f(dval, n);
			delete[] dval;
			for (uint32_t i = 0; i < n; ++i)
				g[i] = r.d[i + 1];
			return r.d[0];
		};
		return find_min(g, x, n);
	}
};