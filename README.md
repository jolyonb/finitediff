# finitediff.py

Finite Difference library by Jolyon Bloomfield and Stephen Face, Copyright 2017

This lightweight library takes 1D finite difference derivatives of data at arbitrary order on a non-uniform grid.

Given a set of x values and a set of y values, the library computes dydx at each x value. It does so by first computing a stencil at the specified order for the given x values. Derivatives of any set of y values can then be computed, without having to recompute the stencil. A derivative can also be extracted at an individual x value without having to compute all derivatives.

Even and odd boundary conditions about x=0 can also be specified, which increases the accuracy of derivatives near those boundaries. These boundary conditions assume that x[0] > 0.

When used with a uniform grid, the stencils are the same as usual finite-difference methods.

Written in Python 3 using numpy, but should also be compatible with Python 2.7.

## Files

* finitediff: The library, example code and testing scripts
* LICENSE: The MIT license under which this library is released
* theory.pdf: A PDF file describing the theory by which the stencils are computed
