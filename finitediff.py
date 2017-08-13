#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library to compute 1D finite difference derivatives on a non-uniform grid

Released under the MIT License

Copyright 2017 Jolyon Bloomfield and Stephen Face
"""

import numpy as np

class DerivativeError(Exception):
    """Generic error raised whenever this library throws an exception"""
    pass

class NoStencil(DerivativeError):
    """
    Error raised whenever a routine is called that requires a stencil to be
    precomputed, but stencil does not exist
    """
    pass

class Derivative(object):
    """
    Computes a finite difference derivative on a non-uniform grid at a given order
    * Use set_x to pass in a set of x values and construct the appropriate stencil
    * Then use dydx to compute derivatives at those x values

    Even/Odd boundary conditions can be specified when constructing
    the stencil initially. Alternatively, a given stencil can be modified to
    change the type of boundary condition later (eg, from even to odd)
    """

    def __init__(self, order):
        """
        Sets the order of the finite difference code
        """
        # Stores the number of grid points in a stencil
        self._N = order + 1
        # Store the number of points to the left to use in the derivative
        # If odd, uses more points to the right
        self._leftpoints = order // 2
        # Initialize a dummy stencil
        self.stencil = np.array([[0.0]])
        # Initialize storage for x values
        self._xvals = None

    def set_x(self, xvals, boundary=0, copy=False):
        """
        Pass in an array of x values
        Computes the weights of a derivative operation for a given set of y
        Allows for different boundary conditions at x=0:
            -1 = odd
            0 = no boundary
            1 = even
        Even and odd boundary conditions assume that the first grid point is
        at x > 0
        Set copy = True to store a copy of the x values
        Else, a reference to the x values is stored instead
        """
        length = len(xvals)
        if length < self._N:
            raise DerivativeError("Grid too short for given order")

        # Store the x values
        if copy:
            self._xvals = xvals.copy()
        else:
            self._xvals = xvals

        # Initialize the stencil
        if np.shape(self.stencil) == (length, length):
            self.stencil.fill(0.0)
        else :
            # Make a new array
            self.stencil = np.zeros([length, length])

        # Do gridpoints on left side first
        self._apply_boundary(boundary, length)

        # Do gridpoints in middle next
        for i in range(self._leftpoints, length - self._N + self._leftpoints + 1):
            # Find the left- and right-most points in the stencil
            # Note that rp is the index of the rightmost point + 1
            lp = i - self._leftpoints   # left point
            rp = lp + self._N           # right point
            self._default_weights(i - lp, xvals[lp:rp], self.stencil[i, lp:rp])

        # Do gridpoints on right side last
        for i in range(length - self._N + self._leftpoints + 1, length):
            lp = length - self._N
            rp = length
            self._default_weights(i - lp, xvals[lp:rp], self.stencil[i, lp:rp])

    def apply_boundary(self, boundary=0):
        """
        Applies a boundary condition to the stencil for the left hand points,
        without recomputing the entire stencil. This facilitates changing
        between odd/even boundary conditions.
        Uses the xvals that were passed into set_x.
        boundary: Allows for different boundary conditions at x=0:
            -1 = odd
            0 = no boundary
            1 = even
        Even and odd boundary conditions assume that the first grid point is
        at x > 0
        """
        # Check that we set_x has been called
        if self._xvals is None:
            raise NoStencil("No stencil has been created yet")
        # Check that the stencil is the correct shape (internal consistency)
        length = len(self._xvals)
        if np.shape(self.stencil) != (length, length):
            raise DerivativeError("Stencil is wrong shape for these x values")
        # Apply the boundary condition
        self._apply_boundary(boundary, length)

    def _apply_boundary(self, boundary, length):
        """
        Internal method for actually applying the boundary condition.
        Assumes error checking has been performed.
        boundary is as in apply_boundary
        length is the length of self._xvals
        """
        # Go and update the stencils for the points on the left
        lp = 0        # left point
        rp = self._N   # right point

        # Which boundary condition will we use?
        if boundary == 0:
            # No boundary condition
            for i in range(self._leftpoints):
                self._default_weights(i - lp, self._xvals[lp:rp], self.stencil[i, lp:rp])
        else:
            # Even/Odd
            for i in range(self._leftpoints):
                self._boundary_weights(i - lp, self._xvals[lp:rp], self.stencil[i, lp:rp], boundary)

    def _default_weights(self, i, xvals, stencil):
        """
        Compute the weights for a given grid point
        No fancy boundary conditions here
        i is the point of interest
        xvals is the slice of data used for the computation
        stencil is where we will save the resulting weights

        Both xvals and stencil are vectors of length N
        """

        for j in range(self._N):
            # We do the i case by adding together all the rest of the results
            # at the end of the computation
            if j == i :
                stencil[j] = 0  # Don't pollute the sum that computes stencil[i]
                continue
            # We compute l'_j(x_i)
            # Start with the denominator
            denom = 1.0
            xdiff = xvals[j] - xvals
            for a in range(self._N):
                if a == j :
                    continue
                denom *= xdiff[a]
            # Now do the numerator
            num = 1.0
            xdiff = xvals[i] - xvals
            for b in range(self._N):
                if b == j or b == i:
                    continue
                num *= xdiff[b]
            # Compute the weight
            stencil[j] = num / denom

        # Add the contribution to the ith component
        stencil[i] = - np.sum(stencil)

    def _boundary_weights(self, i, xvals, stencil, multiplier):
        """
        Compute the weights for a given grid point using even/odd boundary
        conditions at the origin.
        i is the point of interest
        xvals is the slice of data used for the computation
        stencil is where we will save the resulting weights
        multiplier = +1 for even, -1 for odd

        Both xvals and stencil are vectors of length N
        """

        # How many points are off the end with an even/odd boundary?
        delta = self._leftpoints - i
        # Construct a new xvals vector from this data
        newxvals = np.concatenate((-xvals[delta - 1::-1], xvals[0:self._N-delta]))
        # Construct a new i value for newxvals
        newi = i + delta
        # Make a new stencil placeholder
        newstencil = np.zeros_like(stencil)

        # Construct the stencil for newi, newxvals, store in newstencil
        self._default_weights(newi, newxvals, newstencil)

        # If the boundary condition is odd, flip the appropriate signs
        if multiplier == -1:
            for i in range(delta):
                newstencil[i] *= -1

        # Reconstruct the stencil for the original xvals
        stencil.fill(0.0)
        # Copy over the unflipped components
        for i in range(self._N - delta):
            stencil[i] = newstencil[i + delta]
        # Now add in the flipped components
        for i in range(delta):
            stencil[delta - i - 1] += newstencil[i]

    def dydx(self, yvals):
        """
        Pass in a vector of y values
        Returns a vector of dy/dx values
        Must use set_x to construct the stencil first
        """
        if self._xvals is None:
            raise NoStencil("No stencil has been created yet")
        if len(self._xvals) != len(yvals):
            raise DerivativeError("xvals and yvals have different dimensions")
        return np.dot(self.stencil, yvals)

    def leftdydx(self, yvals):
        """
        Pass in a vector of y values
        Returns the derivative at the first position
        Must use set_x to construct the stencil first
        """
        if self._xvals is None:
            raise NoStencil("No stencil has been created yet")
        return np.dot(self.stencil[0], yvals)

    def rightdydx(self, yvals):
        """
        Pass in a vector of y values
        Returns the derivative at the last position
        Must use set_x to construct the stencil first
        """
        if self._xvals is None:
            raise NoStencil("No stencil has been created yet")
        return np.dot(self.stencil[-1], yvals)

    def position_dydx(self, yvals, pos):
        """
        Pass in a vector of y values and a position index (can be negative)
        Returns the derivative at the position given by index
        Must use set_x to construct the stencil first, else an error is raised
        """
        if self._xvals is None:
            raise NoStencil("No stencil has been created yet")
        return np.dot(self.stencil[pos], yvals)

    def get_order(self):
        """Returns the order of the derivative"""
        return self._N - 1

    def get_xvals(self, copy=True):
        """
        Returns the x values that the stencil is made for.
        Set copy = False to get a reference instead of a copy.
        """
        if self._xvals is None:
            raise NoStencil("No stencil has been created yet")
        if copy:
            return self._xvals.copy()
        else:
            return self._xvals
