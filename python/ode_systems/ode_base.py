import numpy as np
import copy
import time
import ctypes
import h5py
import os 

from ode_systems.preprocess.preprocess import make_ode_file,make_RK2_file

class ODEBase(object):
    def __init__(
        self,
        nsteps = 1,
        **kwargs):

        this_dir = __file__
        #/path/to/wind/python/ode_systems
        for iter in range(3):
            this_dir = os.path.split(this_dir)[0]
        self.datadir = os.path.join(this_dir,'data')
        self.h5name = os.path.join(self.datadir,self.cache_fname)
    
        self.n_integration_steps = nsteps

        self.dumpToCDebugInput()

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
        make_RK2_file(self,self.Ntile)

    derivative_suffix = "}\n"
    jacobian_suffix = "}\n"

### CUDA Solvers
    def runIntegratorOutput(
        self,
        integrator_fn,
        integrator_name,
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
                    self.n_integration_steps,
                    constants,equations,
                    self.Nsystems,self.Neqn_p_sys,
                    print_flag = print_flag)]

            walltimes+=[time.time()-init_time]
            tcur+=dt
            times+=[tcur]
            nloops+=1
            equations_over_time[nloops]=copy.copy(equations)

        print('final (tcur=%.2f):'%tcur,np.round(equations_over_time.astype(float),3)[-1][:self.Neqn_p_sys])

        if output_mode is not None:
        
            with h5py.File(self.h5name,output_mode) as handle:
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
                print('nsteps:',nsteps)
        print("total nsteps:",np.sum(nsteps))

    def dumpToCDebugInput(self):
        print("writing:",self.Nsystems,
            "systems",self.Neqn_p_sys,
            "equations per system")
        with open(os.path.join(
            self.datadir,
            self.name+'_debug.txt'),'w') as handle:
            
            handle.write(
                 "float tnow = %s;\n"%str(self.tnow))
            handle.write(
                "float tend = %s;\n" % str(self.tend/self.n_output_steps))
            handle.write(
                "int n_integration_steps = %s;\n" % str(self.n_integration_steps))
            
            fmt_equations = ["%.3e"%val if val != 0 else "0" for val in self.equations.flatten()]
            fmt_equations = ",".join(fmt_equations)
            fmt_equations = '{' + fmt_equations + '}'
            handle.write(
                "float equations[%d] = %s;\n" % (self.Nsystems*self.Neqn_p_sys,fmt_equations))
            handle.write(
                "float new_equations[%d] = %s;\n" % (self.Nsystems*self.Neqn_p_sys,fmt_equations))
            fmt_constants = ["%.3e"%val if val != 0 else "0" for val in self.constants.flatten()]
            fmt_constants = ",".join(fmt_constants)
            fmt_constants = '{' + fmt_constants + '}'
            handle.write(
                "float constants[%d] = %s;\n" % (self.Nsystems*self.nconst,fmt_constants))

            handle.write(
                "int Nsystems = %s;\n" % str(self.Nsystems))

            handle.write(
                "int Neqn_p_sys = %s;\n" % str(self.Neqn_p_sys))

def runCudaIntegrator(
    integrator,
    tnow,tend,
    n_integration_steps,
    constants,equations,
    Nsystems,Nequations_per_system,
    print_flag=1):

    if print_flag:
        print("equations before:",equations)

    before = copy.copy(equations)
    nsteps =integrator(
        ctypes.c_float(np.float32(tnow)),
        ctypes.c_float(np.float32(tend)),
        ctypes.c_int(int(n_integration_steps)),
        constants.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        equations.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(Nsystems),
        ctypes.c_int(Nequations_per_system),
        )

    if print_flag:
        print("equations after %d steps:"%nsteps,equations.astype(np.float32))
        print(tnow,tend)
    return nsteps
