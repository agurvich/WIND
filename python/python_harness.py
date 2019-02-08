import numpy as np 
import ctypes
import time
import os
import copy
import h5py
import time

## find that shared object library 
curdir = os.path.split(os.getcwd())[0]
exec_call = os.path.join(curdir,"cuda","lib","wind.so")
print(exec_call)
c_obj = ctypes.CDLL(exec_call)

#print(dir(c_obj))
#print(c_obj.__dict__)
c_cudaIntegrateEuler = getattr(c_obj,"_Z18cudaIntegrateEulerffPfS_ii")
c_cudaSIE_integrate = getattr(c_obj,"_Z16cudaIntegrateSIEffPfS_ii")
c_cudaBDF2_integrate = getattr(c_obj,"_Z17cudaIntegrateBDF2ffPfS_ii")


def runCudaIntegrator(
    integrator,
    tnow,tend,
    constants,equations,
    Nsystems,Nequations_per_system,
    print_flag=1):

    if print_flag:
        print("equations before:",equations)

    before = copy.copy(equations)
    nsteps =integrator(
        ctypes.c_float(np.float32(tnow)),
        ctypes.c_float(np.float32(tend)),
        constants.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        equations.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(Nsystems),
        ctypes.c_int(Nequations_per_system),
        )

    if print_flag:
        print("equations after %d steps:"%nsteps,equations.astype(np.float32))
        print("residuals:",equations-(before+constants*(tend**3-tnow**3)/3.))
        print(tnow,tend)
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
    max_steps = 25
    tcur = tnow
    dt = (tend-tnow)/max_steps
    equations_over_time = np.zeros((max_steps+1,len(equations)))
    nloops=0
    equations_over_time[nloops]=copy.copy(equations)
    times = []
    times+=[tcur]
    nsteps = []
    walltimes = []

    while nloops < max_steps:#while tcur < tend:
        init_time = time.time()
        nsteps+=[
            runCudaIntegrator(
                integrator_fn,
                tcur,tcur+dt,
                constants,equations,
                Nsystems,Nequations_per_system,
                print_flag = 0)]
        walltimes+=[time.time()-init_time]
        tcur+=dt
        times+=[tcur]
        nloops+=1
        equations_over_time[nloops]=copy.copy(equations)
    print(np.round(equations_over_time.astype(float),3))
    if output_mode is not None:
        with h5py.File("katz96_out.hdf5",output_mode) as handle:
            group = handle.create_group(integrator_name)
            group['equations_over_time'] = equations_over_time
            group['times'] = times
            group['nsteps'] = nsteps
            group['walltimes'] = walltimes
   
####### Test for y' = ct #######
tnow = 0
tend = 2
Nsystems = 2
Nequations_per_system = 5
TEMP = 1e2 ## K
DENSITY = 1e2 ## 1/cm^-3

constants_dict = {}



## From Alex R
#H0: 4.4e-11 s^-1
#He0: 3.7e-12 s^-1
#He+: 1.7e-14

def get_constants(TEMP,Nsystems):
    constants_dict['Gamma_(e,H0)'] = 5.85e-11 * np.sqrt(TEMP)/(1+(TEMP/1e5)) * np.exp(-157809.1/TEMP)
    constants_dict['Gamma_(gamma,H0)'] = 4.4e-11
    constants_dict['alpha_(H+)'] = 8.4e-11 * np.sqrt(TEMP) * (TEMP/1e3)**-0.2 / (1+(TEMP/1e6)**0.7)
    constants_dict['Gamma_(e,He0)'] =  2.38e-11 * np.sqrt(TEMP)/(1+(TEMP/1e5)) * np.exp(-285335.4/TEMP)
    constants_dict['Gamma_(gamma,He0)'] = 3.7e-12
    constants_dict['Gamma_(e,He+)'] =  5.68e-12 * np.sqrt(TEMP)/(1+(TEMP/1e5)) * np.exp(-631515.0/TEMP) 
    constants_dict['Gamma_(gamma,He+)'] = 1.7e-14
    constants_dict['alpha_(He+)'] = 1.5e-10 * TEMP**-0.6353
    constants_dict['alpha_(d)'] =  1.9e-3 * TEMP**-1.5 * np.exp(-470000.0/TEMP) * (1+0.3*np.exp(-94000.0/TEMP))
    constants_dict['alpha_(He++)'] = 3.36e-10 * TEMP**-0.5 * (TEMP/1e3)**-0.2 / (1+(TEMP/1e6)**0.7)

    ## /* constants = [
    ##  Gamma_(e,H0), Gamma_(gamma,H0), 
    ##  alpha_(H+),
    ##  Gamma_(e,He0), Gamma_(gamma,He0),
    ##  Gamma_(e,He+), Gamma_(gamma,He+),
    ##      alpha_(He+), alpha_(d),
    ##  alpha_(He++)
    ##  ] 
    ## */

    constants = np.array([
        constants_dict['Gamma_(e,H0)'],
        constants_dict['Gamma_(gamma,H0)'],
        constants_dict['alpha_(H+)'],
        constants_dict['Gamma_(e,He0)'],
        constants_dict['Gamma_(gamma,He0)'],
        constants_dict['Gamma_(e,He+)'],
        constants_dict['Gamma_(gamma,He+)'],
        constants_dict['alpha_(He+)'],
        constants_dict['alpha_(d)'],
        constants_dict['alpha_(He++)']]*Nsystems)

    ##return np.array(list(range(10))*Nsystems).astype(np.float32)
    return (constants*3.15e7).astype(np.float32) ## convert to 1/yr

def initialize_equations(density,Nsystems):
    return np.array([
    0.25, ## H0
    0.25, ## H+
    0.25, ## He0
    0.125,## He+
    0.125 ## He++
    ]*Nsystems).astype(np.float32)*density

"""
constants = get_constants(TEMP,Nsystems)
equations = initialize_equations(DENSITY,Nsystems)

runIntegratorOutput(
    c_cudaIntegrateEuler,'RK2',
    tnow,tend,
    constants,
    equations,
    Nsystems,
    Nequations_per_system,
    output_mode = 'w')

print("---------------------------------------------------")
"""

constants = get_constants(TEMP,Nsystems)
equations = initialize_equations(DENSITY,Nsystems)

runIntegratorOutput(
    c_cudaSIE_integrate,'SIE',
    tnow,tend,
    constants,
    equations,
    Nsystems,
    Nequations_per_system,
    output_mode = 'w')
#print(constants)

print("---------------------------------------------------")

constants = get_constants(TEMP,Nsystems)
equations = initialize_equations(DENSITY,Nsystems)

runIntegratorOutput(
    c_cudaBDF2_integrate,'BDF2',
    tnow,tend,
    constants,
    equations,
    Nsystems,
    Nequations_per_system,
    output_mode = 'a')

### LEGACY
"""
import h5py
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

