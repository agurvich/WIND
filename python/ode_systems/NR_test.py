import os
import numpy as np 
import time
import os
import copy

## this package imports
from wind.python.ode_systems.ode_base import ODEBase

import wind.python.odecache
import warnings
import h5py

class NR_test(ODEBase):

    def __init__(
        self,
        tend=5,
        **kwargs):

        ## run the ode_base __init__
        super().__init__(
            name='NR_test',
            nconst=4,
            Neqn_p_sys=2,
            tend=tend,
            **kwargs)

        self.eqn_labels = [str.encode('UTF-8') for str in ['u','v']]

        ## make sure that we have implemented the necessary methods
        #self.validate()

    def init_constants(self):
        return np.array([998., 1998.,-999., -1999.]).astype(np.float32)
        
    def init_equations(self):
        return np.array([1.0,0.0]).astype(np.float32)

    def calculate_jacobian(self,system_index=0):
        if type(system_index) == int:
            constants = self.constants[
                system_index*self.nconst:
                (system_index+1)*self.nconst]
        else:
            ## assume we're being passed the current state 
            ##  of this system
            equations,constants = system_index

        jacobian_flat = [
            constants[0], constants[1],
            constants[2], constants[3]]
        return super().calculate_jacobian(jacobian_flat=jacobian_flat)

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

        rates = np.zeros(2)
        rates[0] = 998.*y[0] + 1998.*y[1] # eq. 16.6.1
        rates[1] = -999.*y[0] - 1999.*y[1]

        return super().calculate_derivative(rates=rates)

    def calculate_eqmss(self):
        eqmss = np.tile(np.array([
            2*np.exp(-self.tend) - np.exp(-1000*self.tend),
            -np.exp(-self.tend) + np.exp(-1000*self.tend)
        ]),self.Ntile)
        return eqmss
    
    def dumpToODECache(self):
        with h5py.File(self.h5name,'a') as handle:  
            try:
                group = handle.create_group('Equilibrium')
            except:
                del handle['Equilibrium']
                group = handle.create_group('Equilibrium')
                print("overwriting: Equilibrium")

            group['eqmss'] = self.eqmss.reshape(self.Nsystems,self.Neqn_p_sys)

            ## call the base class method
            super().dumpToODECache(handle)

    def make_plots(self):
        print("Making plots to ../plots")
        this_system = odecache.ODECache(self.h5name)
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
        ridx = lambda x: self.reindex(x,Ntile,self.orig_Neqn_p_sys,this_tile)
        return """
    Jacobian[%d] = constants[0];
    Jacobian[%d] = constants[2];
    Jacobian[%d] = constants[1];
    Jacobian[%d] = constants[3];
"""%(ridx(0),ridx(1),ridx(2),ridx(3))

    def make_derivative_block(self,this_tile,Ntile):
        ridx = lambda x: x+this_tile *self.orig_Neqn_p_sys
        return """
    // eq. 16.6.1 in NR 
    dydt[%d] = constants[0]*equations[%d] + constants[1]*equations[%d];
    dydt[%d] = constants[2]*equations[%d] + constants[3]*equations[%d];
"""%(ridx(0),ridx(0),ridx(1),ridx(1),ridx(0),ridx(1))

    def make_device_derivative_block(self,this_tile,Ntile):
        ridx = lambda x: x+this_tile *self.orig_Neqn_p_sys
        strr = """
    if (threadIdx.x == %d){
        return constants[0]*equations[%d] + constants[1]*equations[%d];
    }"""%(ridx(0),ridx(0),ridx(1))

        strr+="""
    else if (threadIdx.x == %d){
        return constants[2]*equations[%d] + constants[3]*equations[%d];
    }"""%(ridx(1),ridx(0),ridx(1))
        return strr

    ## strings to prepend to calculateDerivative and calculateJacobian functions
    dconstants_string = ""
    jconstants_string = None
