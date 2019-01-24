import numpy as np 
import ctypes
import time
import os
import copy
import h5py

## find that shared object library 
curdir = os.path.split(os.getcwd())[0]
exec_call = os.path.join(curdir,"cuda","lib","wind.so")
c_obj = ctypes.CDLL(exec_call)

#print(dir(c_obj))
#print(c_obj.__dict__)
c_cudaIntegrateEuler = getattr(c_obj,"_Z18cudaIntegrateEulerffPfS_ii")
c_cudaSIE_integrate = getattr(c_obj,"_Z16cudaIntegrateSIEffPfS_ii")

def runCudaIntegrator(tnow,tend,constants,equations,Nsystems,Nequations_per_system,print_flag=1):
    if print_flag:
        print("equations before:",equations)

    before = copy.copy(equations)
    nsteps =c_cudaIntegrateEuler(
        ctypes.c_float(tnow),
        ctypes.c_float(tend),
        constants.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        equations.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(Nsystems),
        ctypes.c_int(Nequations_per_system),
        )

    if print_flag:
        print("equations after %d steps:"%nsteps,equations)
        print("residuals:",equations-(before+constants*(tend**3-tnow**3)/3.))
    return nsteps

def runCudaSIEIntegrator(tnow,tend,constants,equations,Nsystems,Nequations_per_system,print_flag=1):
    if print_flag:
        print("equations before:",equations)

    before = copy.copy(equations)
    nsteps = c_cudaSIE_integrate(
        ctypes.c_float(tnow), ## current time
        ctypes.c_float(tend), ## end time
        constants.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ## current state vector
        equations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ## current state vector
        ctypes.c_int(Nsystems), ## number of systems
        ctypes.c_int(Nequations_per_system), ## number of equations
    )

    if print_flag:
        print("equations after %d steps:"%nsteps,equations)
        ##print("residuals:",equations-constants*((tend)**2-(tnow)**2)/2.)
        print("residuals:",equations-(before+constants*(tend**3-tnow**3)/3.))
    return nsteps

def runIntegratorOutput(
    integrator_fn,
    integrator_name,
    tnow,tend,
    constants,
    equations,
    Nsystems,
    Nequations_per_system,
    output_mode=None):

    ## initialize integration breakdown variables
    tcur = tnow
    dt = (tend-tnow)/10.
    equations_over_time = np.zeros((11,len(equations)))
    nloops=0
    equations_over_time[nloops]=copy.copy(equations)
    times = []
    times+=[tcur]
    nsteps = []

    while tcur < tend:
        nsteps+=[integrator_fn(
        tnow,tcur+dt,
        constants,equations,
        Nsystems,Nequations_per_system,
        print_flag = 0)]

        tcur+=dt
        times+=[tcur]
        nloops+=1
        equations_over_time[nloops]=copy.copy(equations)
    print(equations_over_time)
    if output_mode is not None:
        with h5py.File("cubic_out.hdf5",output_mode) as handle:
            group = handle.create_group(integrator_name)
            group['equations_over_time'] = equations_over_time
            group['times'] = times
            group['nsteps'] = nsteps
   
####### Test for y' = ct #######
tnow = 1.0
tend = 2
Nsystems = 2
Nequations_per_system = 3

constants = np.array([1,2,3,1,2,3]).astype(np.float32)
equations = np.arange(Nsystems*Nequations_per_system).astype(np.float32)
runIntegratorOutput(
    runCudaIntegrator,'Euler',
    tnow,tend,
    constants,
    equations,
    Nsystems,
    Nequations_per_system,
    output_mode = 'w')
print("---------------------------------------------------")
constants = np.array([1,2,3,1,2,3]).astype(np.float32)
equations = np.arange(Nsystems*Nequations_per_system).astype(np.float32)
runIntegratorOutput(
    runCudaSIEIntegrator,'SIE',
    tnow,tend,
    constants,
    equations,
    Nsystems,
    Nequations_per_system,
    output_mode = 'a')

### LEGACY
"""
import h5py
"""


"""
############### SNe Functions ############### 
class SupernovaCluster(ctypes.Structure):
    pass

SupernovaCluster._fields_ = [
                ("xs", ctypes.POINTER(ctypes.c_float)),
                ("ys", ctypes.POINTER(ctypes.c_float)),
                ("zs", ctypes.POINTER(ctypes.c_float)),
                ("ids",ctypes.POINTER(ctypes.c_float)),
                ("launchTimes", ctypes.POINTER(ctypes.c_float)),
                ("coolingTimes", ctypes.POINTER(ctypes.c_float)),
                ("linkingLengths", ctypes.POINTER(ctypes.c_float)),
                ("numNGB",ctypes.c_int),
                ("cluster_id",ctypes.c_int),
                ("NextCluster",ctypes.POINTER(SupernovaCluster))
            ]

"""

