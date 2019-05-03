import sys
import numpy as np
import numpy.linalg as la
from math import *
#from dataio import *

"""
# Function that defines derivatives of y components.
#
# dy1/dt = f1(y, t)
#   ...
# dyN/dt = fN(y, t)
def f(y, t):
  pass

# Function that defines the Jacobian for the system.
#
# Jij = dfi/dyj (partial derivative)
def J(y, t):
  pass
"""

# Integrate using semi-implicit extrapolation method desribed in
# Numerical recipes
#
# y0: vector of initial values
# dt: time-step, assumed fixed
# f_func: function defining dy/dt
# J_func: function defining Jacobian dfi/dyj
#
# Based on NR in C, section 16.6
#
# tend should be an exact multiple of dt to ensure that final value
# is accurate.
def integrate_sie(y0,constants, dt, tend, f_func, J_func):
  # Size of the ODE system.
  N = len(y0)

  # Identity matrix.
  idm = np.identity(N, np.float)
  
  # Special first step.
  Delta0 = np.matmul(la.inv(idm - dt*J_func((y0,constants))), (dt*f_func((y0,constants))))
  y1 = y0 + Delta0

  # Deltakmo = Delta_(k-1)
  yk = y1
  Deltakmo = Delta0

  # Keep track of full solution vs. t.
  t_arr = np.arange(0.0, tend+dt, dt)
  y_arr = np.zeros((len(t_arr), len(y0)))
  
  y_arr[0] = yk

  #t = dt # start after one time-step since y1 computed already
  for i in range(1, len(t_arr)):
    t = t_arr[i]

    # Term involving Jacobian and matrix inverse.
    Jterm = la.inv(idm - dt*J_func((yk,constants)))

    # Term involving f(yk) and Delta_(k-1).
    fterm = dt*f_func((yk,constants)) - Deltakmo

    Deltak = Deltakmo + 2.*np.matmul(Jterm, fterm)

    ykpo = yk + Deltak

    # Keep track of full solution vs. t.
    y_arr[i] = ykpo

    # Increment time.
    t = t + dt

    # For the next step, y_(k+1) -> y_k.
    #                    Deltak -> Delta_(k-1) 
    yk = ykpo
    Deltakmo = Deltak

  # If tend is an exact multiple of dt, then (barring round-off
  # errors), t=tend at this point.

  # Special last (smoothing) step.
  Jterm = la.inv(idm - dt*J_func((yk,constants)))
  fterm = dt*f_func((yk,constants)) - Deltakmo
  
  Deltam = np.matmul(Jterm, fterm) 
  ym_bar = yk + Deltam

  y_arr[len(t_arr)-1] = ym_bar

  return (t_arr, y_arr)

# y derivative (f) for NR test.
def f_NR_test(y,constants=None):
  up = 998.*y[0] + 1998.*y[1] # eq. 16.6.1
  vp = -999.*y[0] - 1999.*y[1]
  return np.array((up, vp))

# Jacobian matrix for NR test.
def J_NR_test(y,constants=None):
  return np.array([[998., 1998.],[-999., -1999.]])

if __name__=="__main__":
  ## Try example still ODE system in NR section 16.6. 
  # y = [u, v]

  y0 = np.array([1., 0.])

  dt = 0.01
  tend = 5.

  (t_arr_sol, y_arr_sol) = integrate_sie(y0, np.zeros(4), dt, tend, f_NR_test, J_NR_test)

  print("t_arr_sol=",t_arr_sol)

  print("y_arr_sol=",y_arr_sol)
