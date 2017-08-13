#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for finite_diff library
"""

import unittest
import finitediff
import random
from math import pi
import numpy as np

class TestFiniteDiff(unittest.TestCase):
    order = 4

    def setUp(self):
        """Initialize a differentiator on a random grid"""
        # Randomly pick some x values
        numvals = 40
        self.x = np.sort(np.array([random.uniform(0.0, 2*pi) for i in range(numvals)]))

        # Create the differentiator on these x values
        self.diff = finitediff.Derivative(TestFiniteDiff.order)
        self.diff.set_x(self.x)

    def test_order(self):
        """Make sure we can get the order out correctly"""
        self.assertEqual(self.diff.get_order(), TestFiniteDiff.order)

    def test_cos(self):
        """Test a cosine function with no boundary conditions"""
        ycos = np.cos(self.x)
        dycos = self.diff.dydx(ycos)
        truevals = -np.sin(self.x)
        self.compare_arrays(dycos, truevals)

    def test_sin(self):
        """Test a sine function with no boundary conditions"""
        ysin = np.sin(self.x)
        dysin = self.diff.dydx(ysin)
        truevals = np.cos(self.x)
        self.compare_arrays(dysin, truevals)

    def test_matrix(self):
        """Test a matrix function with no boundary conditions"""
        ysin = np.sin(self.x)
        ycos = np.cos(self.x)
        test = np.array([ysin, ycos]).transpose()
        truevals = np.array([ycos, -ysin]).transpose()
        dtest = self.diff.dydx(test)
        self.assertTrue(np.all(np.abs(dtest - truevals) < 0.005))

    def test_sin_odd(self):
        """Test a sine function with odd boundary conditions"""
        self.diff.apply_boundary(-1)
        ysin = np.sin(self.x)
        dysin = self.diff.dydx(ysin)
        truevals = np.cos(self.x)
        self.compare_arrays(dysin, truevals)

    def test_cos_even(self):
        """Test a cosine function with even boundary conditions"""
        self.diff.apply_boundary(1)
        ycos = np.cos(self.x)
        dycos = self.diff.dydx(ycos)
        truevals = -np.sin(self.x)
        self.compare_arrays(dycos, truevals)

    def compare_arrays(self, array1, array2):
        """Helper function to test equality of two arrays"""
        for i in range(len(array1)):
            self.assertAlmostEqual(array1[i], array2[i], delta=0.005)

    def test_conversion1(self):
        """Test converting boundary conditions"""
        oldstencil = self.diff.stencil.copy()
        self.diff.set_x(self.x, 1)
        newstencil = self.diff.stencil.copy()
        self.diff.apply_boundary(0)

        # Test converting to no boundary condition
        self.assertTrue(np.all(self.diff.stencil == oldstencil))

        # Test converting to even boundary condition
        newdiff = finitediff.Derivative(TestFiniteDiff.order)
        newdiff.set_x(self.x, 1)
        self.assertTrue(np.all(newdiff.stencil == newstencil))

        # Test converting to odd boundary condition
        self.diff.set_x(self.x, -1)
        newstencil = self.diff.stencil.copy()
        newdiff = finitediff.Derivative(TestFiniteDiff.order)
        newdiff.set_x(self.x, -1)
        self.assertTrue(np.all(newdiff.stencil == newstencil))

    def test_bad(self):
        """Make sure an error is raised appropriately"""
        # Insufficient gridpoints for order
        with self.assertRaises(finitediff.DerivativeError):
            self.diff.set_x(np.array([0.0, 0.5, 1.0]))

        # Something has gone very wrong - stencil and gridpoints are
        # out of alignment
        with self.assertRaises(finitediff.DerivativeError):
            test = finitediff.Derivative(TestFiniteDiff.order)
            test.set_x(np.array([1.0, 2, 3, 4, 5, 6]))
            test._xvals = np.array([1.0, 2.0, 3.0, 4.0])
            test.apply_boundary()

        # Various tests for no stencil
        with self.assertRaises(finitediff.NoStencil):
            test = finitediff.Derivative(TestFiniteDiff.order)
            test.dydx(np.array([1, 2, 3]))

        with self.assertRaises(finitediff.NoStencil):
            test = finitediff.Derivative(TestFiniteDiff.order)
            test.leftdydx(np.array([1, 2, 3]))

        with self.assertRaises(finitediff.NoStencil):
            test = finitediff.Derivative(TestFiniteDiff.order)
            test.rightdydx(np.array([1, 2, 3]))

        with self.assertRaises(finitediff.NoStencil):
            test = finitediff.Derivative(TestFiniteDiff.order)
            test.position_dydx(np.array([1, 2, 3]), 1)

        with self.assertRaises(finitediff.NoStencil):
            test = finitediff.Derivative(TestFiniteDiff.order)
            test.get_xvals()

        with self.assertRaises(finitediff.NoStencil):
            test = finitediff.Derivative(TestFiniteDiff.order)
            test.apply_boundary()

        # xvals and yvals are out of alignment
        with self.assertRaises(finitediff.DerivativeError):
            test = finitediff.Derivative(TestFiniteDiff.order)
            test.set_x(np.array([1.0, 2, 3, 4, 5, 6]))
            test.dydx(np.array([1, 2, 3]))

        with self.assertRaises(finitediff.DerivativeError):
            test = finitediff.Derivative(TestFiniteDiff.order)
            test.set_x(np.array([1.0, 2, 3, 4, 5, 6]))
            test.leftdydx(np.array([1, 2, 3]))

        with self.assertRaises(finitediff.DerivativeError):
            test = finitediff.Derivative(TestFiniteDiff.order)
            test.set_x(np.array([1.0, 2, 3, 4, 5, 6]))
            test.rightdydx(np.array([1, 2, 3]))

        with self.assertRaises(finitediff.DerivativeError):
            test = finitediff.Derivative(TestFiniteDiff.order)
            test.set_x(np.array([1.0, 2, 3, 4, 5, 6]))
            test.position_dydx(np.array([1, 2, 3]), 3)

        # Position out of bounds
        with self.assertRaises(IndexError):
            test = finitediff.Derivative(TestFiniteDiff.order)
            test.set_x(np.array([1.0, 2, 3, 4, 5, 6]))
            test.position_dydx(np.array([1, 2, 3, 4, 5, 6]), 7)

    def test_copy(self):
        """Make sure things copy correctly"""
        # Make sure we get references
        xref = self.diff.get_xvals(False)
        self.x[0] += 1
        self.assertTrue(xref[0] == self.x[0])

        # Make sure we get copies!
        self.diff.set_x(self.x, copy=True)
        x = self.diff.get_xvals(True)
        xref = self.diff.get_xvals(False)
        x[0] += 1
        self.assertFalse(xref[0] == x[0])  # get_xvals returned a copy
        xref[0] += 1
        self.assertFalse(xref[0] == self.x[0])  # set_x stored a copy

    def test_positions(self):
        """Test that derivatives at all positions are correct"""
        ysin = np.sin(self.x)
        dysin = self.diff.dydx(ysin)
        for i in range(len(ysin)):
            self.assertAlmostEqual(dysin[i], self.diff.position_dydx(ysin, i), delta=1e-12)
        self.assertAlmostEqual(dysin[0], self.diff.leftdydx(ysin), delta=1e-12)
        self.assertAlmostEqual(dysin[-1], self.diff.rightdydx(ysin), delta=1e-12)

if __name__ == '__main__':
    unittest.main()
