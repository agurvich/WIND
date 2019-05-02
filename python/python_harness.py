## builtin imports
import numpy as np 
import ctypes
import time
import os
import copy
import h5py
import time

import getopt,sys

## chimes_driver imports
from chimes_driver.utils.table_utils import create_table_grid 
from chimes_driver.driver_config import read_parameters, print_parameters 
from chimes_driver.driver_class import ChimesDriver


home_directory = os.environ['HOME']
## NOTE hardcoded in parameter file...
chimes_parameter_file = "src/CHIMES_repos/chimes-driver/example_parameter_files/eqm_table_wind.param"
chimes_parameter_file = os.path.join(home_directory,chimes_parameter_file)

## this package imports
from eqm_eqns import get_eqm_abundances

## set the solar metallicity 
WIERSMA_ZSOLAR = 0.0129

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


def get_constants_equations_chimes(nH_arr,temperature_arr,init_chem_arr):
    constants = np.array([get_constants(nh,temp,1) for (nh,temp) in zip(nH_arr,temperature_arr)]).flatten()
    ## from chimes_dict
    ##  "HI": 1,"HII": 2,"Hm": 3,"HeI": 4, "HeII": 5,"HeIII": 6,
    equations = np.concatenate([init_chem_arr[:,1:3],init_chem_arr[:,4:7]],axis=1).flatten()
    return constants.astype(np.float32),equations.astype(np.float32)

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
    0.0*y_helium, ## He0
    1.0*y_helium,## He+
    0.0*y_helium ## He++
    ]*Nsystems).astype(np.float32)#*nH####### Test for y' = ct #######

def main(
    tnow = 0,
    tend = 200,
    nsteps = 1,
    RK2 = False,
    SIE = True,
    SIM = True,
    CHIMES = False,
    PY = False,
    TEMP = 1e2, ## K
    nH = 1e2, ## cm^-3
    y_helium = 0.1,
    Nequations_per_system = 5,
    n_output_steps = 20,
    fname=None,
    makeplots=True,
    Ntile = 1
    ):

    ## finish dealing with default arguments
    if fname is None:
        fname = "katz_96.hdf5"
    ## tack on the suffix if it's not there
    if fname[-len(".hdf5"):] != '.hdf5':
        fname+='.hdf5' 
    
    fname = os.path.join("..",'data',fname)
    output_mode = 'a'
    print_flag = False

    ### read the chimes grid
    driver_pars, global_variable_pars, gas_variable_pars = read_parameters(chimes_parameter_file)    

    ## overwrite the driver pars
    driver_pars["driver_mode"] = "noneq_evolution"
    driver_pars['n_iterations'] = n_output_steps

    (nH_arr,temperature_arr,
    metallicity_arr,shieldLength_arr,
    init_chem_arr) = create_table_grid(
        driver_pars,
        global_variable_pars)
    helium_mass_fractions = metallicity_arr[:,1]
    y_heliums = helium_mass_fractions / (4*(1-helium_mass_fractions))

    ## how many different systems are in the grid?
    Nsystems = int(nH_arr.shape[0])

    ## use the grid to create flat arrays of rate coefficients and abundance arrays
    init_constants,init_equations = get_constants_equations_chimes(nH_arr,temperature_arr,init_chem_arr)
    
    ## tile the ICs for each system
    if Ntile > 1:
        init_equations = np.concatenate(
            [np.tile(init_equations[
                i*Nequations_per_system:
                (i+1)*Nequations_per_system],
                Ntile)
            for i in range(Nsystems)])
        Nequations_per_system*=Ntile

    if RK2:
        constants = copy.copy(init_constants)#get_constants(nH,TEMP,Nsystems)
        equations = copy.copy(init_equations)#initialize_equations(nH,Nsystems,y_helium)

        runIntegratorOutput(
            c_cudaIntegrateRK2,'RK2',
            tnow,tend,
            nsteps,
            n_output_steps,
            constants,
            equations,
            Nsystems,
            Nequations_per_system,
            fname,
            output_mode = output_mode,
            print_flag = print_flag)

        print("---------------------------------------------------")
        output_mode = 'a'


    ## initialize cublas lazily
    constants = copy.copy(init_constants)#get_constants(nH,TEMP,Nsystems)
    equations = copy.copy(init_equations)#initialize_equations(nH,Nsystems,y_helium)
    runIntegratorOutput(
            c_cudaSIE_integrate,'SIE',
            tnow,tend,
            nsteps,
            n_output_steps,
            constants,
            equations,
            1,
            1,
            fname,
            output_mode = None,
            print_flag = False)


    if SIE:
        constants = copy.copy(init_constants)#get_constants(nH,TEMP,Nsystems)
        equations = copy.copy(init_equations)#initialize_equations(nH,Nsystems,y_helium)

        runIntegratorOutput(
            c_cudaSIE_integrate,'SIE',
            tnow,tend,
            nsteps,
            n_output_steps,
            constants,
            equations,
            Nsystems,
            Nequations_per_system,
            fname,
            output_mode = output_mode,
            print_flag = print_flag)

        print("---------------------------------------------------")
        output_mode = 'a'

    if SIM:
        constants = copy.copy(init_constants)#get_constants(nH,TEMP,Nsystems)
        equations = copy.copy(init_equations)#initialize_equations(nH,Nsystems,y_helium)

        runIntegratorOutput(
            c_cudaSIM_integrate,'SIM',
            tnow,tend,
            nsteps,
            n_output_steps,
            constants,
            equations,
            Nsystems,
            Nequations_per_system,
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
            group['equations_over_time'] = np.tile(y_arr_sol,Nsystems)
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
        equations_over_time = np.zeros((n_output_steps+1,Nequations_per_system))
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

    ## why 4*y_helium? who can say...
    eqmss = np.array([
        get_eqm_abundances(nH,T,4*y_helium) 
        for (nH,T,y_helium) 
        in zip(nH_arr,temperature_arr,y_heliums)
    ])

    if Ntile > 1:
        eqmss = np.concatenate(
            [np.tile(eqmss[
                i*Nequations_per_system//Ntile:
                (i+1)*Nequations_per_system//Ntile],
                Ntile)
            for i in range(Nsystems)])

    with h5py.File(fname,'a') as handle:
        handle.attrs['Nsystems'] = Nsystems
        handle.attrs['Nequations_per_system'] = Nequations_per_system
        handle.attrs['equation_labels'] = [str.encode('UTF-8') for str in ['H0','H+',"He0","He+","He++"]]
        try:
            group = handle.create_group('Equilibrium')
        except:
            del handle['Equilibrium']
            group = handle.create_group('Equilibrium')
            print("overwriting: Equilibrium")
        group['eqmss'] = eqmss.reshape(Nsystems,Nequations_per_system)
        group['grid_nHs'] = np.log10(nH_arr)
        group['grid_temperatures'] = np.log10(temperature_arr)
        group['grid_solar_metals'] = np.log10(metallicity_arr[:,0]/WIERSMA_ZSOLAR)

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

