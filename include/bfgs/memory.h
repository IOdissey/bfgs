// Copyright © 2022 Alexander Abramenkov. All rights reserved.
// Licensed under the Apache License, Version 2.0.
// https://github.com/IOdissey/bfgs

#pragma once

#include <cstdint>
#include <limits>


// Allocate memory for the DVal.
class Memory
{
private:
	const uint32_t _n;
	double* _beg_ptr = nullptr;
	double* _end_ptr;
	double* _idx_ptr;
	int nn = 0;

public:
	// dim - dimension for automatic derivative.
	// size - maximum number of variables.
	//    if size = 0: Standart version. Slower but uses less memory.
	//    if size > 0: Fast version. It is not very safe, but very easy.
	Memory(uint32_t dim, uint32_t size = 100):
		_n(dim + 1)
	{
		if (size == 0)
			return;
		const uint32_t s = _n * size;
		_beg_ptr = new double[s];
		_end_ptr = _beg_ptr + s;
		_idx_ptr = _beg_ptr;
		for (double* ptr = _idx_ptr; ptr < _end_ptr; ptr += _n)
			*ptr = std::numeric_limits<double>::max();
	}

	~Memory()
	{
		if (_beg_ptr)
			delete[] _beg_ptr;
	}

	inline const uint32_t& n() const
	{
		return _n;
	}

	double* create()
	{
		if (_beg_ptr)
		{
			for (double* ptr = _idx_ptr; ptr < _end_ptr; ptr += _n)
			{
				if (*ptr == std::numeric_limits<double>::max())
				{
					*ptr = 0.0;
					_idx_ptr = ptr + _n;
					return ptr;
				}
			}
			return nullptr;
		}
		else
			return new double[_n];
	}

	void free(double* ptr)
	{
		if (!ptr)
			return;
		// --nn;
		// std::cout << nn << std::endl;
		if (_beg_ptr)
		{
			*ptr = std::numeric_limits<double>::max();
			_idx_ptr = ptr;
		}
		else
			delete[] ptr;
	}
};