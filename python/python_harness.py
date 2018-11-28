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
exec_call = os.path.join(os.getcwd(),"arradd.so")
c_obj = ctypes.CDLL(exec_call)

print(dir(c_obj))
print(c_obj.__dict__)
c_arradd = getattr(c_obj,"_Z6arraddiPiS_S_")

## pass the arrays to be added
Narr = 4
arr_a = np.array(range(Narr),dtype=np.int32,order='C')
arr_b = copy.copy(np.array(range(Narr),dtype=np.int32,order='C')[::-1])
out_pointer = (ctypes.c_int*Narr)()
print(arr_a)

c_arradd(
    ctypes.c_int(Narr),
    arr_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    arr_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    ctypes.byref(out_pointer))

print("Python output:",np.ctypeslib.as_array(out_pointer))
