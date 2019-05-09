import os
import numpy as np 
import time
import os
import copy
import warnings

## chimes_driver imports
from chimes_driver.utils.table_utils import create_table_grid 
from chimes_driver.driver_config import read_parameters, print_parameters 
from chimes_driver.driver_class import ChimesDriver


## this package imports
from ode_systems.ode_base import ODEBase
from ode_systems.preprocess.preprocess import reindex

import odecache

home_directory = os.environ['HOME']

## NOTE hardcoded in parameter file...
chimes_parameter_file = "src/CHIMES_repos/chimes-driver/example_parameter_files/eqm_table_wind.param"
chimes_parameter_file = os.path.join(home_directory,chimes_parameter_file)

## set the solar metallicity 
WIERSMA_ZSOLAR = 0.0129

def get_eqm_abundances(nH,TEMP,y_helium):
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

    for key in constants_dict:
        if 'alpha' in key or '(e' in key:
            constants_dict[key]/=nH


    densities = np.zeros(5)
    ne = nH
    for i in range(1000):
        ne = sub_get_densities(constants_dict,nH,TEMP,densities,ne,y_helium) 
        #if (not i%100):
            #print(densities)
    return densities/nH

def sub_get_densities(constants_dict,nH,TEMP,densities,ne,y_helium):
    """ from eqns 33 - 38 of Katz96"""
    ## H0
    densities[0] = (
        nH * constants_dict['alpha_(H+)']/
        (constants_dict['alpha_(H+)'] + 
            constants_dict['Gamma_(e,H0)'] + 
            constants_dict['Gamma_(gamma,H0)']/ne)
    )

    ## H+
    densities[1] = nH - densities[0]

    ## He+
    densities[3] = (
        nH * y_helium/4  / 
        (1 + (
            (constants_dict['alpha_(He+)'] + constants_dict['alpha_(d)']) / 
            (constants_dict['Gamma_(e,He0)']+constants_dict['Gamma_(gamma,He0)']/ne)
            )
        + (
            (constants_dict['Gamma_(e,He+)']+constants_dict['Gamma_(gamma,He+)']/ne)/
            constants_dict['alpha_(He++)']
            )
        )
    )

    ## He0
    densities[2] = (
        densities[3] * (constants_dict['alpha_(He+)']+constants_dict['alpha_(d)'])/
        (constants_dict['Gamma_(e,He0)']+constants_dict['Gamma_(gamma,He0)']/ne)
    )

    ## He++
    densities[4] = (
        densities[3] *
        (constants_dict['Gamma_(e,He+)']+constants_dict['Gamma_(gamma,He+)']/ne)/
        constants_dict['alpha_(He++)']
    )

    ## ne
    ne = densities[1] + densities[3] + densities[4]*2.0
    return ne

def katz_constants(nH,TEMP,Nsystems):
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

class Katz96(ODEBase):
    def __init__(
        self,
        tnow=0,
        tend=200,
        n_output_steps=20,
        Ntile=1,
        **kwargs):

        self.name='Katz96_%d'%Ntile
        self.Ntile = Ntile

        self.eqn_labels = [str.encode('UTF-8') for str in ['H0','H+',"He0","He+","He++"]]
    
        ## integration time variables
        self.tnow = tnow
        self.tend = tend
        self.n_output_steps = n_output_steps

        ### read the chimes grid
        (self.driver_pars,
        self.global_variable_pars,
        self.gas_variable_pars) = read_parameters(
            chimes_parameter_file)    

        (self.nH_arr,self.temperature_arr,
        self.metallicity_arr,self.shieldLength_arr,
        self.init_chem_arr) = create_table_grid(
            self.driver_pars,
            self.global_variable_pars)

        ## overwrite the driver pars
        self.driver_pars["driver_mode"] = "noneq_evolution"
        self.driver_pars['n_iterations'] = n_output_steps

        ## how many different systems are in the grid?
        
        ## initialize equations and constants
        self.equations = self.init_equations()
        self.constants = self.init_constants()


        self.Neqn_p_sys = 5
        self.orig_Neqn_p_sys = self.Neqn_p_sys
        self.Nsystems = int(
            self.equations.shape[0]//self.Neqn_p_sys)
        self.Neqn_p_sys*=self.Ntile

        self.eqmss = self.calculate_eqmss()

        self.nconst = 10

        ## tile the ICs for each system
        if self.Ntile > 1:
            self.equations = np.concatenate(
                [np.tile(self.equations[
                    i*self.Neqn_p_sys:
                    (i+1)*self.Neqn_p_sys],
                    self.Ntile)
                for i in range(self.Nsystems)])
        
        ## make sure that we have implemented the necessary methods
        self.validate()
            
        ## run the ode_base __init__
        super().__init__(**kwargs)
        
    def init_equations(self):
        helium_mass_fractions = self.metallicity_arr[:,1]
        y_heliums = helium_mass_fractions / (4*(1-helium_mass_fractions))

        ## use the grid to create flat arrays of rate coefficients and abundance arrays
        equations = np.concatenate([self.init_chem_arr[:,1:3],self.init_chem_arr[:,4:7]],axis=1).flatten()
        return equations.astype(np.float32)

    def init_constants(self):
 
        ## use the grid to create flat arrays of rate coefficients and abundance arrays
        constants = np.array([katz_constants(nh,temp,1) for (nh,temp) in zip(self.nH_arr,self.temperature_arr)]).flatten()
        return constants.astype(np.float32)
    
    def calculate_jacobian(self,system_index=0):
        if type(system_index) == int:
            constants = self.constants[
                system_index*self.nconst:
                (system_index+1)*self.nconst]

            equations = self.equations[
                system_index*self.Neqn_p_sys:
                (system_index+1)*self.Neqn_p_sys]
        else:
            ## assume we're being passed the current state 
            ##  of this system
            equations,constants = system_index

        if self.Ntile > 1:
            warnings.warn("This python Jacobian function is not tileable")

        ne = equations[1]+equations[3]+equations[4]*2.0;

        ## initialize a flattened jacobian to match C code
        jacobian_flat = np.zeros(self.Neqn_p_sys**2)

        ## H0
        jacobian_flat[0] = -(constants[0]*ne + constants[1]); ## H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
        jacobian_flat[1] = -jacobian_flat[0]; ## H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
        ##H+
        jacobian_flat[6] = -constants[2]*ne; ## H+ -alpha_(H+)ne
        jacobian_flat[5] = -jacobian_flat[6]; ## H0 : 2-alpha_(H+)ne
        
        ## He0
        jacobian_flat[12] = -(constants[3]*ne+constants[4]); ##He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
        jacobian_flat[13] = jacobian_flat[12]; ##He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
        ## He+
        jacobian_flat[19] = constants[5]*ne+constants[6]; ##He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
        jacobian_flat[17] = (constants[7]+constants[8])*ne; ##He0 : (7-alpha_(He+)+8-alpha_(d))ne
        jacobian_flat[18] = -jacobian_flat[17] - jacobian_flat[19]; ##He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
        ## He++
        jacobian_flat[24] = -constants[9]*ne;##He++ : -alpha_(He++)ne
        jacobian_flat[23] = -jacobian_flat[24];##He+ : 9-alpha_(He++)ne
        
        ## indices above are in column major order to match cuBLAS specification
        return jacobian_flat.reshape(self.Neqn_p_sys,self.Neqn_p_sys).T

    def calculate_derivative(self,system_index=0):
        if type(system_index) == int:
            equations = self.equations[
                system_index*self.Neqn_p_sys:
                (system_index+1)*self.Neqn_p_sys]

            constants = self.constants[
                system_index*self.nconst:
                (system_index+1)*self.nconst]
        
        else:
            ## assume we're being passed the current state 
            ##  of this system
            equations,constants = system_index

        rates = np.zeros(5)

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

    def calculate_eqmss(self):
        helium_mass_fractions = self.metallicity_arr[:,1]
        y_heliums = helium_mass_fractions / (4*(1-helium_mass_fractions))

        ## why 4*y_helium? who can say...
        eqmss =  np.array([
            get_eqm_abundances(nH,T,4*y_helium) 
            for (nH,T,y_helium) 
            in zip(self.nH_arr,self.temperature_arr,y_heliums)])

        if self.Ntile > 1:
            llist = []
            for i in range(self.Nsystems):
                foo = np.tile(
                    eqmss[
                        i*self.Neqn_p_sys//self.Ntile:
                        (i+1)*self.Neqn_p_sys//self.Ntile],self.Ntile)
                llist +=[foo]
            eqmss = np.concatenate(llist)
        return eqmss


    def dumpToODECache(self,group=None):
        if group is None:
            return

        group['eqmss'] = self.eqmss.reshape(self.Nsystems,self.Neqn_p_sys)
        group['grid_nHs'] = np.tile(np.log10(self.nH_arr),self.Nsystem_tile)
        group['grid_temperatures'] = np.tile(np.log10(self.temperature_arr),self.Nsystem_tile)
        group['grid_solar_metals'] = np.tile(np.log10(self.metallicity_arr[:,0]/WIERSMA_ZSOLAR),self.Nsystem_tile)

    def make_plots(self):
        print("Making plots to ../plots")
        this_system = odecache.ODECache(
            self.name,
            self.h5name)

        this_system.plot_all_systems(
            subtitle = None,
            plot_eqm = True,
            savefig = '../plots/%s.pdf'%self.name,
            xlow=0,ylow=-0.1,
            yname = '$n_X/n_\mathrm{H}$',
            xname = 't (yrs)',
            )

### PRECOMPILE STUFF FOR MAKING .CU FILES
    def make_jacobian_block(self,this_tile,Ntile):
        ridx = lambda x: reindex(x,Ntile,self.orig_Neqn_p_sys,this_tile)
        jacobian_good_stuff = ("""   
    // H0
    this_block_jacobian[%d] = -(constants[%d]*ne + constants[%d]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[%d] = -this_block_jacobian[%d]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        """%(ridx(0),0,1,ridx(1),ridx(0)) + 
        """
    //H+
    this_block_jacobian[%d] = -constants[%d]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[%d] = -this_block_jacobian[%d]; // H0 : 2-alpha_(H+)ne
        """%(ridx(6),2,ridx(5),ridx(6)) + 
        """
    // He0
    this_block_jacobian[%d] = -(constants[%d]*ne+constants[%d]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[%d] = this_block_jacobian[%d]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        """%(ridx(12),3,4,ridx(13),ridx(12))  + 
        """
    // He+
    this_block_jacobian[%d] = constants[%d]*ne+constants[%d]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[%d] = (constants[%d]+constants[%d])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[%d] = -this_block_jacobian[%d] - 
        this_block_jacobian[%d]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        """%(ridx(19),5,6,ridx(17),7,8,ridx(18),ridx(17),ridx(19)) + 
        """
    // He++
    this_block_jacobian[%d] = -constants[%d]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[%d] = -this_block_jacobian[%d];//He+ : 9-alpha_(He++)ne
        """%(ridx(24),9,ridx(23),ridx(24)))
        return jacobian_good_stuff

    def make_derivative_block(self,this_tile,Ntile):
        ridx = lambda x: x+this_tile *self.orig_Neqn_p_sys
        derivative_good_stuff = ("""    // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[%d] = constants[%d]*ne*this_block_state[%d]
        -(constants[%d]*ne + constants[%d])*this_block_state[%d]; 
        """%(ridx(0),2,ridx(1),0,1,ridx(0)) )
        derivative_good_stuff+=(
        """
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[%d] = -this_block_derivatives[%d];
        """%(ridx(1),ridx(0)))
        derivative_good_stuff+=(
        """
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[%d] = (constants[%d]+constants[%d])*ne*this_block_state[%d] 
        - (constants[%d]*ne+constants[%d])*this_block_state[%d];
        """%(ridx(2),7,8,ridx(3),3,4,ridx(2)))
        derivative_good_stuff+=(
        """
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[%d] = constants[%d]*ne*this_block_state[%d] 
        + (constants[%d]*ne+constants[%d])*this_block_state[%d]  
        - (constants[%d]+constants[%d])*ne*this_block_state[%d] 
        - (constants[%d]*ne+constants[%d])*this_block_state[%d];
        """%(ridx(3),9,ridx(4),3,4,ridx(2),7,8,ridx(3),5,6,ridx(3)))
        derivative_good_stuff+=(
        """
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[%d] = (constants[%d]*ne+constants[%d])*this_block_state[%d]
        -constants[%d]*ne*this_block_state[%d]; 
        """%(ridx(4),5,6,ridx(3),9,ridx(4)))
        return derivative_good_stuff
    
    def make_RK2_block(self,this_tile,Ntile):
        ridx = lambda x: x+this_tile *self.orig_Neqn_p_sys

        strr = """
    if (threadIdx.x == %d){
        // H0 : alpha_(H+) ne nH+ - (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0
        return constants[2]*ne*equations[%d]
            -(constants[0]*ne + constants[1])*equations[%d]; 
    }"""%(ridx(0),ridx(1),ridx(0))

        strr+="""
    else if (threadIdx.x == %d){
        // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
        return -constants[2]*ne*equations[%d]
            +(constants[0]*ne + constants[1])*equations[%d]; 
    }"""%(ridx(1),ridx(1),ridx(0))

        strr+="""
    else if (threadIdx.x == %d){
        // He0 :(alpha_(He+)+alpha_(d)) ne nHe+ - (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
        return (constants[7]+constants[8])*ne*equations[%d] 
            - (constants[3]*ne+constants[4])*equations[%d];
    }"""%(ridx(2),ridx(3),ridx(2))

        strr+="""
    else if (threadIdx.x == %d){
        // He+ : 
        //  alpha_(He++) ne nHe++ 
        //  + (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
        //  - (alpha_(He+)+alpha_(d)) ne nHe+ 
        //  - (Gamma_(e,He+)ne + Gamma_(gamma,He+)) nHe+
        return constants[9]*ne*equations[%d] 
            + (constants[3]*ne+constants[4])*equations[%d]  
            - (constants[7]+constants[8])*ne*equations[%d] 
            - (constants[5]*ne+constants[6])*equations[%d];
    }"""%(ridx(3),ridx(4),ridx(2),ridx(3),ridx(3))

        strr+="""
    else if (threadIdx.x == %d){
        // He++ : -alpha_(He++) ne nHe++
        return -constants[9]*ne*equations[%d];
    }
    """%(ridx(4),ridx(4))

        return strr

    derivative_prefix = """__global__ void calculateDerivatives(
    float * d_derivatives_flat, 
    float * constants, 
    float * equations,
    int Nsystems,
    int Neqn_p_sys,
    float time){
    // isolate this system 

    int bid = get_system_index();
    // don't need to do anything, no system corresponds to this thread-block
    if (bid >= Nsystems){
        return;
    }

    int eqn_offset = bid*Neqn_p_sys;
    float * this_block_state = equations+eqn_offset;
    float * this_block_derivatives = d_derivatives_flat+eqn_offset;

    // constraint equation, ne = nH+ + nHe+ + 2*nHe++
    float ne = equations[eqn_offset+1]+equations[eqn_offset+3]+equations[eqn_offset+4]*2.0;

    /* constants = [
        0-Gamma_(e,H0), 1-Gamma_(gamma,H0), 
        2-alpha_(H+),
        3-Gamma_(e,He0), 4-Gamma_(gamma,He0), 
        5-Gamma_(e,He+), 6-Gamma_(gamma,He+),
        7-alpha_(He+),
        8-alpha_(d),
        9-alpha_(He++)
        ] 
    */
"""

    jacobian_prefix = """__global__ void calculateJacobians(
    float **d_Jacobianss, 
    float * constants,
    float * equations,
    int Nsystems,
    int Neqn_p_sys,
    float time){

    // isolate this system 
    int bid = get_system_index();

    // don't need to do anything, no system corresponds to this thread-block
    if (bid >= Nsystems){
        return;
    }

    int eqn_offset = bid*Neqn_p_sys;
    float * this_block_state = equations+eqn_offset;
    float * this_block_jacobian = d_Jacobianss[bid];

    // constraint equation, ne = nH+ + nHe+ + 2*nHe++
    float ne = this_block_state[1]+this_block_state[3]+this_block_state[4]*2.0;

    /* constants = [
        0-Gamma_(e,H0), 1-Gamma_(gamma,H0), 
        2-alpha_(H+),
        3-Gamma_(e,He0), 4-Gamma_(gamma,He0), 
        5-Gamma_(e,He+), 6-Gamma_(gamma,He+),
        7-alpha_(He+),
        8-alpha_(d),
        9-alpha_(He++)
        ] 
    */

"""

    jacobian_suffix = "}\n"

    RK2_prefix = """
#include <stdio.h>
#include <math.h>

#include "explicit_solver.h"

__device__ float calculate_dydt(
    float tnow,
    float * constants,
    float * equations){
    // constraint equation, ne = nH+ + nHe+ + 2*nHe++
    float ne = equations[1]+equations[3]+equations[4]*2.0;

    /* constants = [
        Gamma_(e,H0), Gamma_(gamma,H0), 
        alpha_(H+),
        Gamma_(e,He0), Gamma_(gamma,He0), 
        Gamma_(e,He+), Gamma_(gamma,He+),
        alpha_(He+),
        alpha_(d),
        alpha_(He++)
        ] 
    */ 
"""

    RK2_suffix = """
   else{
        return NULL;
    } 
} // calculate_dydt\n"""
