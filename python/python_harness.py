import numpy as np 
import ctypes
import time
import os
import copy

## find that shared object library 
curdir = os.path.split(os.getcwd())[0]
exec_call = os.path.join(curdir,"cuda","arradd.so")
c_obj = ctypes.CDLL(exec_call)

#print(dir(c_obj))
#print(c_obj.__dict__)
c_cudaIntegrateEuler = getattr(c_obj,"_Z18cudaIntegrateEulerffPfS_ii")
c_cudaSIE_integrate = getattr(c_obj,"_Z13SIE_integrateiiffPf")

def runCudaIntegrator(tnow,timestep,constants,equations,Nsystems,Nequations_per_system,print_flag=1):
    if print_flag:
        print("equations before:",equations)

    c_cudaIntegrateEuler(
        ctypes.c_float(tnow),
        ctypes.c_float(timestep),
        constants.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        equations.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(Nsystems),
        ctypes.c_int(Nequations_per_system))

    if print_flag:
        print("equations after:",equations)
        print("residuals:",equations-constants*((tnow+timestep)**2-(tnow)**2)/2.)

def runCudaSIEIntegrator(tnow,timestep,constants,equations,Nsystems,Nequations_per_system,print_flag=1):
    if print_flag:
        print("equations before:",equations)

    c_cudaSIE_integrate(
        ctypes.c_int(Nsystems), ## number of systems
        ctypes.c_int(Nequations_per_system), ## number of equations
        ctypes.c_float(tnow), ## current time
        ctypes.c_float(tnow+timestep), ## end time
        constants.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ## current state vector
        equations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ## current state vector
    )

    if print_flag:
        print("equations after:",equations)
        print("residuals:",equations-constants*((tnow+timestep)**2-(tnow)**2)/2.)
   
####### Test for y' = ct #######
tnow = 0.0
timestep = 1
Nsystems = 2
Nequations_per_system = 3

constants = (np.arange(Nsystems*Nequations_per_system)+1.0).astype(np.float32)
equations = np.arange(Nsystems*Nequations_per_system)

runCudaIntegrator(tnow,timestep,constants,equations,Nsystems,Nequations_per_system)
runCudaSIEIntegrator(tnow,timestep,constants,equations,Nsystems,Nequations_per_system)

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

