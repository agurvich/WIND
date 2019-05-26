import os
import numpy as np 
import time
import os
import copy
import warnings
import h5py

## this package imports
from ode_systems.ode_base import ODEBase

import odecache

class StiffTrig(ODEBase):
    def __init__(
        self,
        tend=6,
        **kwargs):

        ## run the ode_base __init__
        super().__init__(
            name = 'StiffTrig',
            nconst=2,
            Neqn_p_sys = 2,  
            tend=tend,
            **kwargs)
        
        ## what is the name of each equation
        ##  for any plot labels or printing
        self.eqn_labels = [
            str.encode('UTF-8') for str in 
            ['y0',"y1"]]

        ## make sure that we have implemented the necessary methods
        #self.validate()            

    def init_equations(self):
        ## use the grid to create flat arrays of rate coefficients and abundance arrays
        equations = np.array([0,1])
        return equations.astype(np.float32)

    def init_constants(self):
        ## use the grid to create flat arrays of rate coefficients and abundance arrays
        freq0=3
        constants = np.array([
            2*np.pi*freq0,
            2*np.pi*freq0
            ])
        return constants.astype(np.float32)
    
    def calculate_jacobian(
        self,
        tnow,
        equations,
        constants,
        system_index=0):

        jacobian = np.array([
            [-1, constants[1]],
            [-constants[0],-1]])
        jacobian_flat = jacobian.flatten()
        return super().calculate_jacobian(jacobian_flat=jacobian_flat)

    def calculate_derivative(
        self,
        tnow,
        equations,
        constants,
        system_index = 0):

        rates = np.zeros(2)
        rates[0] = -equations[0] + constants[1]*equations[1]
        rates[1] = -equations[1] + constants[0]*equations[0]
        
        return super().calculate_derivative(rates=rates)

    def calculate_eqmss(self):
        eqmss = np.array([0,0])
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
            group['freqs'] = self.constants/(2*np.pi)

            ## call the base class method
            super().dumpToODECache(handle)

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
            yname = 'y',
            xname = 't',
            )

### PRECOMPILE STUFF FOR MAKING .CU FILES
    def make_jacobian_block(self,this_tile,Ntile):
        ridx = lambda x: self.reindex(x,Ntile,self.orig_Neqn_p_sys,this_tile)
        jacobian_good_stuff = ("""   
    // y0
    Jacobian[%d] = -1;
    Jacobian[%d] = constants[%d];
        """%(ridx(0),ridx(2),1) + 
        """
    // y1
    Jacobian[%d] = -constants[%d];
    Jacobian[%d] = -1;
        """%(ridx(1),0,ridx(3)))
        return jacobian_good_stuff

    def make_derivative_block(self,this_tile,Ntile):
        ridx = lambda x: x+this_tile *self.orig_Neqn_p_sys
        derivative_good_stuff = ("""
    // y0
    dydt[%d] = -equations[%d] + constants[%d]*equations[%d];
    // y1
    dydt[%d] = -equations[%d] - constants[%d]*equations[%d];
        """%(ridx(0),ridx(0),1,ridx(1),
            ridx(1), ridx(1),0,ridx(0)) )
        return derivative_good_stuff
    
    def make_device_derivative_block(self,this_tile,Ntile):
        ridx = lambda x: x+this_tile *self.orig_Neqn_p_sys

        strr = """
    if (threadIdx.x == %d){
        return -equations[%d] + constants[%d]*equations[%d]; 
    }"""%(ridx(0),ridx(0),1,ridx(1))
        strr += """
    if (threadIdx.x == %d){
        return -equations[%d] - constants[%d]*equations[%d]; 
    }"""%(ridx(1),ridx(1),0,ridx(0))
        return strr

    dconstants_string = """"""

    ## copy the dconstants string
    jconstants_string = None
