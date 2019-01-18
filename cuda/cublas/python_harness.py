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
exec_call = os.path.join(curdir,'cublas',"invert_test.so")
c_obj = ctypes.CDLL(exec_call)

#print(dir(c_obj))
#print(c_obj.__dict__)
c_cudaInvertMatrix = getattr(c_obj,"_Z12invertMatrixiPfi")


def runCudaInvertMatrix(arr_a,arr_b):

    joined = np.append(arr_a,arr_b).astype(np.float32)
    c_cudaInvertMatrix(
        ctypes.c_int(2),
        joined.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(int(arr_a.shape[0]**0.5))
    )

    A = arr_a.reshape(3,3)
    B = arr_b.reshape(3,3)
    Ainv = joined[:arr_a.shape[0]].reshape(3,3)
    Binv = joined[arr_a.shape[0]:].reshape(3,3)
    
    id = np.identity(3)

    print("A:\n",id-A)
    print("A^-1:\n",Ainv)
    print("A^-1 A\n",np.round(np.dot(Ainv,id-A),2))
    print('----')
    print("B:\n",id-B)
    print("B^-1:\n",Binv)
    print("B^-1 B\n",np.round(np.dot(Binv,id-B),2))

    

    

def runCudaIntegrator(tnow,timestep,constants,equations,Nsystems,Nequations_per_system):
    constants = constants.astype(np.float32)
    equations = equations.astype(np.float32)
    
    print("equations before:",equations)
    c_cudaIntegrateEuler(
        ctypes.c_float(tnow),
        ctypes.c_float(timestep),
        constants.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        equations.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(Nsystems),
        ctypes.c_int(Nequations_per_system))

    print("equations after:",equations)
    print("residuals:",equations-constants*timestep**3/3.)

    
####### Test for y' = c ########
tnow = 0.0
timestep = 1
Nsystems = 2
Nequations_per_system = 3

A = np.arange(9).reshape(3,3).astype(np.float32)
B = 2*np.arange(9).reshape(3,3).astype(np.float32)


A = np.array([
    0,3,4,
    1,5,6,
    9,8,2 ]).astype(np.float32)
B = np.array([
    22,3,4,
    1,5,6,
    9,8,2]).astype(np.float32)

#runCudaIntegrator(tnow,timestep,constants,equations,Nsystems,Nequations_per_system)
runCudaInvertMatrix(A.flatten(),B.flatten())
####### Test for y' = ct ########
