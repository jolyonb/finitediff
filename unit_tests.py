#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for finite_diff library
"""

import unittest
import finite_diff
import random
from math import pi
import numpy as np

class TestFiniteDiff(unittest.TestCase):

    def setUp(self):
        """Initialize a differentiator on a random grid"""
        # Randomly pick some x values
        numvals = 40
        self.x = np.sort(np.array([random.uniform(0.0, 2*pi) for i in range(numvals)]))

        # Create the differentiator on these x values
        self.diff = finite_diff.Derivative(5)
        self.diff.set_x(self.x)

    def test_order(self):
        """Make sure we can get the order out correctly"""
        self.assertEqual(self.diff.get_order(), 5)

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

    def test_sin_odd(self):
        """Test a sine function with odd boundary conditions"""
        self.diff.apply_boundary(self.x, -1)
        ysin = np.sin(self.x)
        dysin = self.diff.dydx(ysin)
        truevals = np.cos(self.x)
        self.compare_arrays(dysin, truevals)

    def test_cos_even(self):
        """Test a cosine function with even boundary conditions"""
        self.diff.apply_boundary(self.x, 1)
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
        self.diff.apply_boundary(self.x, 0)

        # Test converting to no boundary condition
        self.assertTrue(np.all(self.diff.stencil == oldstencil))

        # Test converting to even boundary condition
        newdiff = finite_diff.Derivative(5)
        newdiff.set_x(self.x, 1)
        self.assertTrue(np.all(newdiff.stencil == newstencil))

        # Test converting to odd boundary condition
        self.diff.set_x(self.x, -1)
        newstencil = self.diff.stencil.copy()
        newdiff = finite_diff.Derivative(5)
        newdiff.set_x(self.x, -1)
        self.assertTrue(np.all(newdiff.stencil == newstencil))

    def test_bad(self):
        """Make sure an error is raised appropriately"""
        with self.assertRaises(finite_diff.DerivativeError):
            self.diff.set_x(np.array([0.0, 0.5, 1.0]))

        with self.assertRaises(finite_diff.DerivativeError):
            self.diff.apply_boundary(self.x[:-1])

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
