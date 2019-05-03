import numpy as np
import copy
import time
import ctypes

from ode_systems.preprocess.preprocess import make_ode_file

class ODEBase(object):
    pass

    def validate(self):
        self.init_constants()
        self.init_equations()
        self.calculate_jacobian()
        self.calculate_derivative()
        self.calculate_eqmss()
        self.dumpToODECache()

    def init_constants(self):
        raise NotImplementedError
        
    def init_equations(self):
        raise NotImplementedError

    def calculate_jacobian(self):
        raise NotImplementedError

    def calculate_derivative(self):
        raise NotImplementedError

    def calculate_eqmss():
        raise NotImplementedError

    def dumpToODECache(self):
        raise NotImplementedError

    def preprocess(self):
        make_ode_file(self,self.Ntile)


### CUDA Solvers
    def runIntegratorOutput(
        self,
        integrator_fn,
        integrator_name,
        n_integration_steps,
        output_mode=None,
        print_flag = 0):

        equations = copy.copy(self.equations)
        constants = copy.copy(self.constants)

        ## initialize integration breakdown variables
        tcur = self.tnow
        dt = (self.tend-self.tnow)/self.n_output_steps
        equations_over_time = np.zeros((self.n_output_steps+1,len(equations)))
        nloops=0
        equations_over_time[nloops]=copy.copy(equations)
        times = []
        times+=[tcur]
        nsteps = []
        walltimes = []

        while nloops < self.n_output_steps:#while tcur < tend:
            init_time = time.time()
            nsteps+=[
                runCudaIntegrator(
                    integrator_fn,
                    tcur,tcur+dt,
                    n_integration_steps,
                    constants,equations,
                    self.Nsystems,self.Neqn_p_sys,
                    print_flag = print_flag)]

            walltimes+=[time.time()-init_time]
            tcur+=dt
            times+=[tcur]
            nloops+=1
            equations_over_time[nloops]=copy.copy(equations)

        print('final (tcur=%.2f):'%tcur,np.round(equations_over_time.astype(float),3)[-1][:5])

        if output_mode is not None:
            with h5py.File(self.cache_fname,output_mode) as handle:
                try:
                    group = handle.create_group(integrator_name)
                except:
                    del handle[integrator_name]
                    group = handle.create_group(integrator_name)
                    print("overwriting:",integrator_name)
                group['equations_over_time'] = equations_over_time
                group['times'] = times
                group['nsteps'] = nsteps
                group['walltimes'] = walltimes
                print(walltimes,'walls')
        print("total nsteps:",np.sum(nsteps))

def runCudaIntegrator(
    integrator,
    tnow,tend,
    nsteps,
    constants,equations,
    Nsystems,Nequations_per_system,
    print_flag=1):

    if print_flag:
        print("equations before:",equations)

    before = copy.copy(equations)
    nsteps =integrator(
        ctypes.c_float(np.float32(tnow)),
        ctypes.c_float(np.float32(tend)),
        ctypes.c_int(int(nsteps)),
        constants.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        equations.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(Nsystems),
        ctypes.c_int(Nequations_per_system),
        )

    if print_flag:
        print("equations after %d steps:"%nsteps,equations.astype(np.float32))
        print(tnow,tend)
    return nsteps
