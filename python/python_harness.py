import numpy as np 
import ctypes
import time
import os
import copy
import h5py
import time

import getopt,sys

from eqm_eqns import get_eqm_abundances


## find that shared object library 
curdir = os.path.split(os.getcwd())[0]
exec_call = os.path.join(curdir,"cuda","lib","wind.so")
print(exec_call)
c_obj = ctypes.CDLL(exec_call)

#print(dir(c_obj))
#print(c_obj.__dict__)
c_cudaIntegrateRK2 = getattr(c_obj,"_Z16cudaIntegrateRK2ffPfS_ii")
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
    output_mode=None,
    print_flag = 0):

    ## initialize integration breakdown variables
    max_steps = 20#tend-tnow
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
                print_flag = print_flag)]
        walltimes+=[time.time()-init_time]
        tcur+=dt
        times+=[tcur]
        nloops+=1
        equations_over_time[nloops]=copy.copy(equations)
    print('final (tcur=%.2f):'%tcur,np.round(equations_over_time.astype(float),3)[-1][:5])
    if output_mode is not None:
        with h5py.File("katz96_out.hdf5",output_mode) as handle:
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
    print("total nsteps:",np.sum(nsteps))
   
def calculate_rates(equations,constants):
    rates = np.zeros(5)
    equations = equations[:len(equations)//2]
    constants = constants[:len(constants)//2]

    ne = equations[1] + equations[3] + 2*equations[4]
    ## H0 : alpha_(H+) ne nH+ - (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0
    rates[0] = (constants[2]*ne*equations[1]
        -(constants[0]*ne + constants[1])*equations[0]) 
    ## H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    rates[1] = (-constants[2]*ne*equations[1]
            +(constants[0]*ne + constants[1])*equations[0]) 
    ## He0 :(alpha_(He+)+alpha_(d)) ne nHe+ - (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
    rates[2] = ((constants[7]+constants[8])*ne*equations[3] 
            - (constants[3]*ne+constants[4])*equations[2])
    ## He+ : 
    ##  alpha_(He++) ne nHe++ 
    ##  + (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
    ##  - (alpha_(He+)+alpha_(d)) ne nHe+ 
    ##  - (Gamma_(e,He+)ne + Gamma_(gamma,He+)) nHe+
    rates[3] = (constants[9]*ne*equations[4] 
            + (constants[3]*ne+constants[4])*equations[2]  
            - (constants[7]+constants[8])*ne*equations[3] 
            - (constants[5]*ne+constants[6])*equations[3])
    ## He++ : -alpha_(He++) ne nHe++
    rates[4] = (-constants[9]*ne*equations[4])

    return rates

def get_constants(nH,TEMP,Nsystems):
  ## From Alex R
    #H0: 4.4e-11 s^-1
    #He0: 3.7e-12 s^-1
    #He+: 1.7e-14

    constants_dict = {}
    constants_dict['Gamma_(e,H0)'] = 5.85e-11 * np.sqrt(TEMP)/(1+np.sqrt(TEMP/1e5)) * np.exp(-157809.1/TEMP) * nH
    constants_dict['Gamma_(gamma,H0)'] = 4.4e-11
    constants_dict['alpha_(H+)'] = 8.4e-11 / np.sqrt(TEMP) * (TEMP/1e3)**-0.2 / (1+(TEMP/1e6)**0.7) * nH
    constants_dict['Gamma_(e,He0)'] =  2.38e-11 * np.sqrt(TEMP)/(1+np.sqrt(TEMP/1e5)) * np.exp(-285335.4/TEMP) * nH
    constants_dict['Gamma_(gamma,He0)'] = 3.7e-12
    constants_dict['Gamma_(e,He+)'] =  5.68e-12 * np.sqrt(TEMP)/(1+np.sqrt(TEMP/1e5)) * np.exp(-631515.0/TEMP) * nH
    constants_dict['Gamma_(gamma,He+)'] = 1.7e-14
    constants_dict['alpha_(He+)'] = 1.5e-10 * TEMP**-0.6353 * nH
    constants_dict['alpha_(d)'] =  (
        1.9e-3 * TEMP**-1.5 * 
        np.exp(-470000.0/TEMP) * 
        (1+0.3*np.exp(-94000.0/TEMP)) * nH )
    constants_dict['alpha_(He++)'] = 3.36e-10 * TEMP**-0.5 * (TEMP/1e3)**-0.2 / (1+(TEMP/1e6)**0.7) * nH

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

    return (constants*3.15e7).astype(np.float32) ## convert to 1/yr

def initialize_equations(nH,Nsystems,y_helium):
    return np.array([
    0.0, ## H0
    1.0, ## H+
    0.0*y_helium/4., ## He0
    1.0*y_helium/4.,## He+
    0.0*y_helium/4. ## He++
    ]*Nsystems).astype(np.float32)#*nH####### Test for y' = ct #######


def main(
    tnow = 0,
    tend = 200,
    Nsystems = 100,
    RK2 = False,
    SIE = True,
    BDF2 = False,
    TEMP = 1e2, ## K
    nH = 1e2, ## cm^-3
    y_helium = 0.4,
    Nequations_per_system = 5,
    ):
    
    Nsystems = int(Nsystems)

    output_mode = 'a'
    print_flag = False

    if RK2:
        constants = get_constants(nH,TEMP,Nsystems)
        equations = initialize_equations(nH,Nsystems,y_helium)

        runIntegratorOutput(
            c_cudaIntegrateRK2,'RK2',
            tnow,tend,
            constants,
            equations,
            Nsystems,
            Nequations_per_system,
            output_mode = output_mode,
            print_flag = print_flag)

        print("---------------------------------------------------")
        output_mode = 'a'

    if SIE:
        constants = get_constants(nH,TEMP,Nsystems)
        equations = initialize_equations(nH,Nsystems,y_helium)

        runIntegratorOutput(
            c_cudaSIE_integrate,'SIE',
            tnow,tend,
            constants,
            equations,
            Nsystems,
            Nequations_per_system,
            output_mode = output_mode,
            print_flag = print_flag)

        print("---------------------------------------------------")
        output_mdoe = 'a'

    if BDF2:
        constants = get_constants(nH,TEMP,Nsystems)
        equations = initialize_equations(nH,Nsystems,y_helium)

        runIntegratorOutput(
            c_cudaBDF2_integrate,'BDF2',
            tnow,tend,
            constants,
            equations,
            Nsystems,
            Nequations_per_system,
            output_mode = output_mode,
            print_flag = print_flag)

    eqm_abundances = get_eqm_abundances(nH,TEMP,y_helium)
    print('eqm:',[float('%.3f'%abundance) for abundance in eqm_abundances])
    with h5py.File("katz96_out.hdf5",'a') as handle:
        group = handle.attrs['eqm_abundances'] = eqm_abundances

    print("Rates:")
    print(calculate_rates(equations,constants))

if __name__ == '__main__':
    argv = sys.argv[1:]
    opts,args = getopt.getopt(argv,'',['tnow=','tend=','Nsystems=','RK2=','SIE=','BDF2='])
    #options:
    #--snap(low/high) : snapshot numbers to loop through
    #--savename : name of galaxy to use
    #--mps : mps flag, default = 0
    for i,opt in enumerate(opts):
        if opt[1]=='':
            opts[i]=('mode',opt[0].replace('-',''))
        else:
            try:
                ## if it's an int or a float this should work
                opts[i]=(opt[0].replace('-',''),eval(opt[1]))
            except:
                ## if it's a string... not so much
                opts[i]=(opt[0].replace('-',''),opt[1])
    main(**dict(opts))


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

