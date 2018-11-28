import numpy as np 
import ctypes
import time
import os
import copy
"""
import h5py
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

## find that shared object library 
curdir = os.path.split(os.getcwd())[0]
exec_call = os.path.join(curdir,"cuda","arradd.so")
c_obj = ctypes.CDLL(exec_call)

print(dir(c_obj))
print(c_obj.__dict__)
c_cudaIntegrateRiemann = getattr(c_obj,"_Z20cudaIntegrateRiemannffPfS_ii")

## pass the arrays to be added
Narr = 4
arr_a = np.array(range(Narr),dtype=np.int32,order='C')
arr_b = copy.copy(np.array(range(Narr),dtype=np.int32,order='C')[::-1])
out_pointer = (ctypes.c_int*Narr)()
print(arr_a)

tnow = 0.0
timestep = 0.5
Nsystems = 1
Nequations_per_system = 1 

constants = np.ones(Nsystems*Nequations_per_system,dtype=np.float32)
equations = np.zeros(Nsystems*Nequations_per_system,dtype=np.float32)

print("equations before:",equations)
c_cudaIntegrateRiemann(
    ctypes.c_float(tnow),
    ctypes.c_float(timestep),
    constants.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    equations.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    ctypes.c_int(Nsystems),
    ctypes.c_int(Nequations_per_system))

print("equations after:",equations)
