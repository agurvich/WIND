import numpy as np 
import copy

def integrate_rk2(
    tnow,
    tend,
    timestep,
    equations,
    constants,
    f_func,
    Neqn_p_sys,
    absolute,
    relative,
    DEBUG=None):

    equations = np.array(equations,dtype=np.float64)
    y1 = np.zeros(Neqn_p_sys)
    y2 = np.zeros(Neqn_p_sys)

    while tnow < tend:
        timestep = min(timestep,tend-tnow)
        y1 = copy.copy(equations)
        y2 = copy.copy(equations)
        #import pdb; pdb.set_trace()

        rk2_step(
            tnow,
            tnow+timestep,
            1,
            y1,
            constants,
            f_func)

        rk2_step(
            tnow,
            tnow+timestep,
            2,
            y2,
            constants,
            f_func)

        if checkError(y1,y2,absolute,relative):
            timestep/=2
        else:
            equations = 2*y2-y1
            if DEBUG is not None:
                DEBUG.append((tnow,copy.copy(equations)))
            tnow+=timestep
            timestep*=2#(tend-tnow)
            
    return equations

def rk2_step(
    tnow,
    tend,
    n_integration_steps,
    equations,
    constants,
    f_func):

    timestep = (tend-tnow)/n_integration_steps
    for step_i in range(n_integration_steps):  
        dydts = f_func(tnow,equations,constants)
        equations+=dydts*timestep

    return equations
    
def checkError(y1,y2,absolute,relative):
    delta = y2-y1
    if np.any(np.abs(delta) > absolute):
        return True
    ## have to handle individually
    for y1_i,y2_i in zip(y1,y2):
        if ( y1_i > absolute and 
                y2_i > absolute and 
                np.abs((y2_i-y1_i)/(2*y2_i-y1_i+1e-12)) > relative):
            return True
    return False

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
