## builtin imports
import numpy as np 
import ctypes
import time
import os
import copy
import h5py

import getopt,sys

from ode_systems.katz96 import Katz96

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

def runCudaIntegrator(
    integrator,
    tnow,tend,
    nsteps,
    constants,equations,
    Nsystems,Nequations_per_system,
    print_flag=1):

    if print_flag:
        print("equations before:",equations)

    before = copy.copy(equations)
    nsteps =integrator(
        ctypes.c_float(np.float32(tnow)),
        ctypes.c_float(np.float32(tend)),
        ctypes.c_int(int(nsteps)),
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
    n_integration_steps,
    n_output_steps,
    constants,
    equations,
    Nsystems,
    Nequations_per_system,
    fname,
    output_mode=None,
    print_flag = 0):

    ## initialize integration breakdown variables
    tcur = tnow
    dt = (tend-tnow)/n_output_steps
    equations_over_time = np.zeros((n_output_steps+1,len(equations)))
    nloops=0
    equations_over_time[nloops]=copy.copy(equations)
    times = []
    times+=[tcur]
    nsteps = []
    walltimes = []

    while nloops < n_output_steps:#while tcur < tend:
        init_time = time.time()
        nsteps+=[
            runCudaIntegrator(
                integrator_fn,
                tcur,tcur+dt,
                n_integration_steps,
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
        with h5py.File(fname,output_mode) as handle:
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
            print(walltimes,'walls')
    print("total nsteps:",np.sum(nsteps))
   
def main(
    tnow = 0,
    tend = 200,
    nsteps = 1,
    RK2 = False,
    SIE = True,
    SIM = True,
    CHIMES = False,
    PY = False,
    Nequations_per_system = 5,
    n_output_steps = 20,
    fname=None,
    makeplots=True,
    NR = False,
    katz = True,
    **kwargs):

    if katz:
        system = Katz96(
            tnow,
            tend,
            n_output_steps,
            **kwargs)

    elif NR:
        system = NR(
            tnow,
            tend,
            n_output_steps,
            **kwargs)

    ## finish dealing with default arguments
    if fname is None:
        fname = "katz_96.hdf5"
    ## tack on the suffix if it's not there
    if fname[-len(".hdf5"):] != '.hdf5':
        fname+='.hdf5' 
    
    fname = os.path.join("..",'data',fname)
    output_mode = 'a'
    print_flag = False
 
    init_equations,init_constants = system.equations,system.constants

    if RK2:
        constants = copy.copy(init_constants)
        equations = copy.copy(init_equations)

        runIntegratorOutput(
            c_cudaIntegrateRK2,'RK2',
            tnow,tend,
            nsteps,
            n_output_steps,
            constants,
            equations,
            system.Nsystems,
            system.Neqn_p_sys,
            fname,
            output_mode = output_mode,
            print_flag = print_flag)

        print("---------------------------------------------------")
        output_mode = 'a'

    ## initialize cublas lazily
    constants = copy.copy(init_constants)
    equations = copy.copy(init_equations)
    runIntegratorOutput(
            c_cudaSIE_integrate,'SIE',
            tnow,tnow+1e-12,
            1,
            n_output_steps,
            constants,
            equations,
            1,
            1,
            fname,
            output_mode = None,
            print_flag = False)

    if SIE:
        constants = copy.copy(init_constants)
        equations = copy.copy(init_equations)

        runIntegratorOutput(
            c_cudaSIE_integrate,'SIE',
            tnow,tend,
            nsteps,
            n_output_steps,
            constants,
            equations,
            system.Nsystems,
            system.Neqn_p_sys,
            fname,
            output_mode = output_mode,
            print_flag = print_flag)

        print("---------------------------------------------------")
        output_mode = 'a'

    if SIM:
        constants = copy.copy(init_constants)
        equations = copy.copy(init_equations)

        runIntegratorOutput(
            c_cudaSIM_integrate,'SIM',
            tnow,tend,
            nsteps,
            n_output_steps,
            constants,
            equations,
            system.Nsystems,
            system.Neqn_p_sys,
            fname,
            output_mode = output_mode,
            print_flag = print_flag)

    if PY:
        raise UnimplementedError("SIE not implemented for Katz96 yet")
        import sie
        y0 = np.array([1., 0.])

        dt = 0.01
        tend = 5.

        init = time.time()
        (t_arr_sol, y_arr_sol) = sie.integrate_sie(y0, dt, tend, sie.f_NR_test, sie.J_NR_test)
        wall = time.time() - init
        nsteps = tend/dt
        with h5py.File(fname,output_mode) as handle:
            if 'PY' in handle.keys():
                del handle['PY']
                print("Overwriting PY")
            group = handle.create_group('PY')
            ## not memory efficient but it will jive with ODECache at least
            group['equations_over_time'] = np.tile(y_arr_sol,system.Nsystems)
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
        equations_over_time = np.zeros((n_output_steps+1,system.Neqn_p_sys))
        times = np.linspace(tnow,tend,n_output_steps+1,endpoint=True)
        nsteps = np.zeros(n_output_steps) ## no way to measure this :[ 

        ## change the DT within chimes-driver
        my_driver.myGasVars.hydro_timestep = (tend - tnow)*3.15e7/n_output_steps ## s

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
            with h5py.File(fname,output_mode) as handle:
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

    with h5py.File(fname,'a') as handle:
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
        import odecache
        print("Making plots to ../plots")
        this_system = odecache.ODECache(fname)
        this_system.plot_all_systems(
            subtitle = None,
            plot_eqm = True,
            savefig = '../plots/Katz96_out.pdf')

if __name__ == '__main__':
    argv = sys.argv[1:]
    opts,args = getopt.getopt(argv,'',[
        'tnow=','tend=',
        'nsteps=',
        'n_output_steps=',
        'Nsystems=',
        'RK2=','SIE=','SIM=',
        'PY=','CHIMES=',
        'fname=','makeplots=',
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

