# finitediff.py

[![Build Status](https://travis-ci.org/jolyonb/finitediff.svg?branch=master)](https://travis-ci.org/jolyonb/finitediff)
[![Coverage Status](https://coveralls.io/repos/github/jolyonb/finitediff/badge.svg?branch=master)](https://coveralls.io/github/jolyonb/finitediff?branch=master)
![Python Versions](https://img.shields.io/badge/python-2.7%2C%203.4%2C%203.5%2C%203.6-blue.svg)

Finite Difference library by Jolyon Bloomfield and Stephen Face, Copyright 2017

This lightweight library takes 1D finite difference derivatives of data at arbitrary order on a non-uniform grid, built using numpy arrays.

Given a set of x values and a set of y values, the library computes dydx at each x value. It does so by first computing a stencil at the specified order for the given x values. Derivatives of any set of y values can then be computed, without having to recompute the stencil. A derivative can also be extracted at an individual x value without having to compute all derivatives.

Even and odd boundary conditions about x=0 can also be specified, which increases the accuracy of derivatives near those boundaries. These boundary conditions assume that x[0] > 0.

When used with a uniform grid, the stencils are the same as usual finite-difference methods.

The y values may even be matrix-valued. So long as the first index of yvals is the gridpoint, matrix derivatives will be computed correctly.

## Files

* finitediff: The library, example code and testing scripts
* LICENSE: The MIT license under which this library is released
* theory.pdf: A PDF file describing the theory by which the stencils are computed
* example.py: An example of using the library to compute derivatives
