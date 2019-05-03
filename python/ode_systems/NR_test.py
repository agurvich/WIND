import os
import numpy as np 
import time
import os
import copy

## this package imports
from ode_systems.ode_base import ODEBase
from ode_systems.preprocess.preprocess import reindex

import odecache
import warnings

class NR_test(ODEBase):

    def __init__(
        self,
        tnow=0,
        tend=5,
        n_output_steps=50,
        Ntile=1):

        self.name='NR_test'
        self.Ntile = Ntile
    
        self.cache_fname = self.name+'_%d.hdf5'%Ntile

        self.eqn_labels = [str.encode('UTF-8') for str in ['u','v']]
    
        ## integration time variables
        self.tnow = tnow
        self.tend = tend
        self.n_output_steps = n_output_steps 


        self.Neqn_p_sys = 2
        self.Nsystems = 1

        ## initialize equations and constants
        self.equations = self.init_equations()
        self.constants = self.init_constants()

        self.nconst = 4

        ## tile the ICs for each system
        if self.Ntile > 1:
            self.equations = np.concatenate(
                [np.tile(self.equations[
                    i*self.Neqn_p_sys:
                    (i+1)*self.Neqn_p_sys],
                    self.Ntile)
                for i in range(self.Nsystems)])

        self.orig_Neqn_p_sys = self.Neqn_p_sys
        self.Neqn_p_sys*=self.Ntile
        
        ## make sure that we have implemented the necessary methods
        self.validate()

    def init_constants(self):
        return np.tile([998., 1998.,-999., -1999.],self.Nsystems).astype(np.float32)
        
    def init_equations(self):
        return np.tile([1.0,0.0],self.Nsystems).astype(np.float32)

    def calculate_jacobian(self,system_index=0):
        if type(system_index) == int:
            constants = self.constants[
                system_index*self.nconst:
                (system_index+1)*self.nconst]
        else:
            ## assume we're being passed the current state 
            ##  of this system
            equations,constants = system_index

        if self.Ntile > 1:
            warnings.warn("This python Jacobian function is not tileable")

        return np.array([
            [constants[0], constants[1]],
            [constants[2], constants[3]]])

    def calculate_derivative(self,system_index=0):
        if type(system_index) == int:
            y = self.equations[
                system_index*self.Neqn_p_sys:
                (system_index+1)*self.Neqn_p_sys]
        else:
            equations,constants = system_index
            ## assume we're being passed the current state 
            ##  of this system
            y = equations

        up = 998.*y[0] + 1998.*y[1] # eq. 16.6.1
        vp = -999.*y[0] - 1999.*y[1]
        return np.tile(np.array((up, vp)),self.Ntile)

    def calculate_eqmss(self):
        eqmss = np.tile(np.array([
            2*np.exp(-self.tend) - np.exp(-1000*self.tend),
            -np.exp(-self.tend) + np.exp(-1000*self.tend)
        ]),self.Ntile)
        return eqmss
    
    def dumpToODECache(self,group=None):
        if group is None:
            return

        eqmss = self.calculate_eqmss()

        group['eqmss'] = eqmss.reshape(self.Nsystems,self.Neqn_p_sys)

    def make_plots(self):
        print("Making plots to ../plots")
        this_system = odecache.ODECache(self.cache_fname)
        this_system.plot_all_systems(
            subtitle = None,
            plot_eqm = True,
            savefig = '../plots/%s.pdf'%self.name,
            #xlow=0,ylow=-0.1,
            yname = '',
            xname = 't',
            )

### PRECOMPILE STUFF FOR MAKING .CU FILES
    def make_jacobian_block(self,this_tile,Ntile):
        ridx = lambda x: reindex(x,Ntile,self.orig_Neqn_p_sys,this_tile)
        ## TODO make this self consistent
        #constants = np.tile([998., 1998.,-999., -1999.],self.Nsystems).astype(np.float32)
        return """
    this_block_jacobian[%d] = 998.0;
    this_block_jacobian[%d] = -999.0;
    this_block_jacobian[%d] = 1998.0;
    this_block_jacobian[%d] = -1999.0;
"""%(ridx(0),ridx(1),ridx(2),ridx(3))

    def get_derivative_block(self,this_tile,Ntile):
        ridx = lambda x: x+this_tile *self.orig_Neqn_p_sys
        return """
    // eq. 16.6.1 in NR 
    this_block_derivatives[%d] = 998.0*this_block_state[%d] + 1998. * this_block_state[%d];
    this_block_derivatives[%d] = -999.0*this_block_state[%d] - 1999.0*this_block_state[%d];
"""%(ridx(0),ridx(0),ridx(1),ridx(1),ridx(0),ridx(1))

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
"""
