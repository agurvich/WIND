## builtin imports
import numpy as np 
import ctypes
import time
import os
import copy
import h5py

import getopt,sys

from ode_systems.katz96 import Katz96
from ode_systems.NR_test import NR_test
from pysolvers.sie import integrate_sie

## find the first order solver shared object library 
curdir = os.path.split(os.getcwd())[0]
exec_call = os.path.join(curdir,"cuda","lib","sie.so")
print(exec_call)
c_obj = ctypes.CDLL(exec_call)
c_cudaIntegrateRK2 = getattr(c_obj,"_Z16cudaIntegrateRK2ffiPfS_ii")
c_cudaSIE_integrate = getattr(c_obj,"_Z16cudaIntegrateSIEffiPfS_ii")

## get the second order library
exec_call = os.path.join(curdir,"cuda","lib","sie2.so")
c_obj = ctypes.CDLL(exec_call)
c_cudaSIM_integrate = getattr(c_obj,"_Z16cudaIntegrateSIEffiPfS_ii")

def main(
    nsteps = 1,
    RK2 = False,
    SIE = True,
    SIM = False,
    CHIMES = False,
    PY = False,
    makeplots=True,
    NR = True,
    katz = False,
    **kwargs):

    if NR:
        system = NR_test(**kwargs)

    elif katz:
        system = Katz96(**kwargs)
    else:
        raise Exception("pick katz or NR")
 
    output_mode = 'a'
    print_flag = False
 
    init_equations,init_constants = system.equations,system.constants

    if RK2:
        constants = copy.copy(init_constants)
        equations = copy.copy(init_equations)

        system.runIntegratorOutput(
            c_cudaIntegrateRK2,'RK2',
            nsteps,
            output_mode = output_mode,
            print_flag = print_flag)

        print("---------------------------------------------------")
        output_mode = 'a'

    ## initialize cublas lazily
    system.runIntegratorOutput(
        c_cudaSIE_integrate,'SIE',
        1,
        output_mode = None,
        print_flag = False)

    if SIE:
        constants = copy.copy(init_constants)
        equations = copy.copy(init_equations)

        system.runIntegratorOutput(
            c_cudaSIE_integrate,'SIE',
            nsteps,
            output_mode = output_mode,
            print_flag = print_flag)

        print("---------------------------------------------------")
        output_mode = 'a'

    if SIM:
        constants = copy.copy(init_constants)
        equations = copy.copy(init_equations)

        system.runIntegratorOutput(
            c_cudaSIM_integrate,'SIM',
            nsteps,
            output_mode = output_mode,
            print_flag = print_flag)

    if PY:
        yss = []
        for system_index in range(system.Nsystems):
            this_equations = system.equations[
                system_index*system.Neqn_p_sys:
                (system_index+1)*system.Neqn_p_sys]

            this_constants = system.constants[
                system_index*system.nconst:
                (system_index+1)*system.nconst]

            dt = (system.tend-system.tnow)/system.n_output_steps

            init = time.time()
            (t_arr_sol, y_arr_sol) = integrate_sie(
                this_equations, 
                this_constants,
                dt,
                system.tend, 
                system.calculate_derivative,
                system.calculate_jacobian)

            yss+=[y_arr_sol]

        print(np.shape(yss))
        yss = np.array(yss)
        yss = np.transpose(yss,axes=(1,0,2))
        wall = time.time() - init
        nsteps = system.tend/dt
        with h5py.File(system.h5name,output_mode) as handle:
            if 'PY' in handle.keys():
                del handle['PY']
                print("Overwriting PY")
            group = handle.create_group('PY')
            ## not memory efficient but it will jive with ODECache at least
            group['equations_over_time'] = yss
            group['times'] = t_arr_sol
            group['nsteps'] = [nsteps]
            group['walltimes'] = [wall]

    if CHIMES:
        my_driver = ChimesDriver(
            nH_arr, temperature_arr, metallicity_arr, shieldLength_arr, 
            init_chem_arr, 
            driver_pars, global_variable_pars, gas_variable_pars,
            rank = 0)

        ## initialize the output array
        equations_over_time = np.zeros((system.n_output_steps+1,system.Neqn_p_sys))
        times = np.linspace(system.tnow,system.tend,system.n_output_steps+1,endpoint=True)
        nsteps = np.zeros(system.n_output_steps) ## no way to measure this :[ 

        ## change the DT within chimes-driver
        my_driver.myGasVars.hydro_timestep = (system.tend - system.tnow)*3.15e7/system.n_output_steps ## s

        my_driver.walltimes = []
        final_output_array, chimes_cumulative_time = my_driver.run()
        
        equations_over_time = np.transpose(
            np.concatenate(
                np.concatenate(
                    [final_output_array[:,1:3,:],final_output_array[:,4:7,:]]
                    ,axis=1) ## get rid of primordial molecular abundances
                ,axis=0) ## flatten the different systems into one array
            ) ## swap the time and systems axes to match wind convention

        ## output to the savefile
        integrator_name = 'CHIMES'
        if output_mode is not None:
            with h5py.File(system.h5name,output_mode) as handle:
                try:
                    group = handle.create_group(integrator_name)
                except:
                    del handle[integrator_name]
                    group = handle.create_group(integrator_name)
                    print("overwriting:",integrator_name)
                group['equations_over_time'] = equations_over_time
                group['times'] = times
                group['nsteps'] = nsteps
                group['walltimes'] = my_driver.walltimes

        output_mode = 'a'

    with h5py.File(system.h5name,'a') as handle:
        handle.attrs['Nsystems'] = system.Nsystems
        handle.attrs['Nequations_per_system'] = system.Neqn_p_sys
        handle.attrs['equation_labels'] = system.eqn_labels
        try:
            group = handle.create_group('Equilibrium')
        except:
            del handle['Equilibrium']
            group = handle.create_group('Equilibrium')
            print("overwriting: Equilibrium")
        
        ## dump equilibrium to group and system config info
        system.dumpToODECache(group)
        
    if makeplots:
        system.make_plots()
        

if __name__ == '__main__':
    argv = sys.argv[1:]
    opts,args = getopt.getopt(argv,'',[
        'tnow=','tend=',
        'nsteps=',
        'n_output_steps=',
        'Nsystems=',
        'RK2=','SIE=','SIM=',
        'PY=','CHIMES=',
        'NR=','katz=',
        'cache_fname=','makeplots=',
        'Ntile='])

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

