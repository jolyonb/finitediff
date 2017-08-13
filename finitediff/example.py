#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example code for taking derivatives using the finite_diff library
"""

import random
import numpy as np
from math import pi
from finitediff import Derivative

numvals = 40

# Randomly pick some x values
x = np.sort(np.array([random.uniform(0.0, 2*pi) for i in range(numvals)]))

# Take some trig functions
ysin = np.sin(x)
ycos = np.cos(x)

# Initialize differentiator
diff = Derivative(4)
diff.set_x(x)

# Take derivatives with no boundary conditions
dycos = diff.dydx(ycos)
dysin = diff.dydx(ysin)

# Take derivatives with boundary conditions
diff.apply_boundary(1)
dycos2 = diff.dydx(ycos)
diff.apply_boundary(-1)
dysin2 = diff.dydx(ysin)

# How did we go?
print("x", "Actual", "No boundary", "Boundary", "Error 1", "Error2")
for i in range(numvals) :
    print(x[i], ycos[i], dysin[i], dysin2[i], dysin[i] - ycos[i], dysin2[i] - ycos[i])
    print(x[i], -ysin[i], dycos[i], dycos2[i], dycos[i] + ysin[i], dycos2[i] + ysin[i])

# Construct a vector for y
diff.apply_boundary(0)
test = np.array([ysin, ycos]).transpose()
truevals = np.array([ycos, -ysin]).transpose()
dtest = diff.dydx(test)
print("Vector-valued y test:", np.all(np.abs(dtest - truevals) < 0.005))
