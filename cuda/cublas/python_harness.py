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
curdir = os.path.split(os.getcwd())
exec_call = os.path.join(*curdir,"invert_test.so")
print("executing:",exec_call)
c_obj = ctypes.CDLL(exec_call)

#print(dir(c_obj))
#print(c_obj.__dict__)
c_cudaSIE_integrate = getattr(c_obj,"_Z16cudaIntegrateSIEffPfS_ii")


def runCudaSIEIntegrator(nsystems,neqn,current_time,end_time,state,print_flag=1):
    if print_flag:
        print("Current time: %.2f"%current_time)
        print(state)
    c_cudaSIE_integrate(
        ctypes.c_float(current_time), ## current time
        ctypes.c_float(end_time), ## end time
        state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ## current state vector
        state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ## current state vector
        ctypes.c_int(nsystems), ## number of systems
        ctypes.c_int(neqn), ## number of equations
    )

    if print_flag:
        print("Current time: %.2f"%end_time)
        print(state)
    

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
    print("residuals:",equations-constants*timestep**2/2.)

    
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

state = np.arange(6).astype(np.float32)
copy_state =copy.copy(state)
#runCudaIntegrator(tnow,timestep,constants,equations,Nsystems,Nequations_per_system)
current_time = 0.0
end_time = 1.0
runCudaSIEIntegrator(2,3,current_time,end_time,state,False)
predicted_state = copy_state + 0.5*np.array([1,2,3,1,2,3])*(end_time**2 - current_time**2)
print(predicted_state,'predicted')
print(state,'actual')
print((predicted_state-state).astype(np.float32),'residual')
####### Test for y' = ct ########
