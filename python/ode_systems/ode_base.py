import numpy as np
import copy
import time
import ctypes
import h5py

import os 
import sys

cuda_dir = os.path.realpath(__file__)
for i in range(3):
    cuda_dir = os.path.split(cuda_dir)[0]
cuda_dir = os.path.join(cuda_dir,'cuda')
#print("WIND cuda directory:",cuda_dir)

class Precompiler(object):
    def splitODEFile(self,ode_file=None):
        strr = ""

        ode_file = 'ode_system.cu' if ode_file is None else ode_file
        words = []
        with open(
            os.path.join(
                cuda_dir,'ode_system',ode_file)
            ) as handle:
           for line_i,line in enumerate(handle.readlines()):
                if "FLAG FOR PYTHON FRONTEND" in line:
                    words+=[strr]
                    strr=""
                else:
                    strr+=line
        words+=[strr]

        self.derivative_prefix = words[0]
        self.derivative_suffix = ""#words[1]
        self.jacobian_prefix = words[2]
        self.jacobian_suffix = words[4]

    def setPrefixes(self,
            dconstants_string,
            ode_file=None,
            jconstants_string=None):
        if jconstants_string is None:
            jconstants_string = dconstants_string
    
        ## open the file from disk and set the prefix string
        self.splitODEFile(ode_file)
        
        ## add the string that should only be called once but should
        ##  vary from system to system
        self.derivative_prefix += dconstants_string
        self.jacobian_prefix += jconstants_string

    def make_ode_file(
        self,
        proto_file,
        Ntile,
        dconstants_string,
        jconstants_string):

        if 'device' in proto_file:
            derivative_fn = self.make_device_derivative_block
            jconstants_string = ""
        else:
            derivative_fn = self.make_derivative_block

        ## read the prefixes/suffixes for this proto file
        self.setPrefixes(dconstants_string,proto_file,jconstants_string)

        strr=self.make_string(
            derivative_fn,
            Ntile,
            self.derivative_prefix,
            self.derivative_suffix)

        strr+=self.make_string(
            self.make_jacobian_block,
            Ntile,
            self.jacobian_prefix,
            self.jacobian_suffix)

        with open(os.path.join(self.datadir,'precompile_'+proto_file),'w') as handle:
            handle.write(strr)

        return strr
        
    def reindex(self,index,Ntile,Ndim,this_tile):
        nrow = index // Ndim
        return index + (Ntile-1)*Ndim*nrow + (Ndim*Ndim*Ntile+Ndim)*this_tile

    def make_string(self,loop_fn,Ntile,prefix,suffix):
        strr = prefix
        for this_tile in range(Ntile):
            strr+=loop_fn(this_tile,Ntile)
        return strr + suffix

    
    def make_RK2_file(self,Ntile):
        strr = ''
        strr += self.make_string(
            self.make_RK2_block,
            Ntile,
            self.RK2_prefix,
            self.RK2_suffix)


        ## read the file suffix straight
        ##  from source
        with open(
            os.path.join(
                cuda_dir,
                'RK2',
                'kernel_suffix.cu'),
            'r') as handle:

            for line in handle.readlines():
                strr+=line
        
        ## write out to the precompile directory
        with open(
            os.path.join(
                self.datadir,
                'preprocess_RK2_kernel.cu'),'w') as handle:
            handle.write(strr)

class ODEBase(Precompiler):
    def __init__(
        self,
        name,
        nconst,
        Neqn_p_sys,

        tnow = 0,
        tend=1,

        n_integration_steps = 1,
        n_output_steps=20,

        Nsystem_tile=1,
        Ntile=1,
        dumpDebug=False,

        absolute=5e-3,
        relative=5e-3,
        **kwargs):
        if len(kwargs):
            raise KeyError("Unused keys:",list(kwargs.keys()))

        self.name = name

        self.Neqn_p_sys = Neqn_p_sys
        self.nconst = nconst

        ## integration time variables
        self.tnow = tnow
        self.tend = tend

        self.n_integration_steps = n_integration_steps
        self.n_output_steps = n_output_steps

        self.Nsystem_tile = Nsystem_tile
        self.Ntile = Ntile

        self.absolute = absolute
        self.relative = relative

        ## format the absolute and relative tolerances
        abs_string = ("%.0e"%self.absolute).replace('e-0','e')
        rel_string = ("%.0e"%self.relative).replace('e-0','e')

        self.name = '_'.join(
            [self.name,
            'neqntile.%s'%str(self.Ntile),
            'nsystemtile.%s'%str(Nsystem_tile),
            'fixed.%s'%str(n_integration_steps),
            'abs.%s'%abs_string,
            'rel.%s'%rel_string
            ])
        print("created:",self.name)

        ## now that we have our name figure out where we're saving our
        ##  data to
        this_dir = __file__
        #/path/to/wind/python/ode_systems
        for iter in range(3):
            this_dir = os.path.split(this_dir)[0]
        self.datadir = os.path.join(this_dir,'data',self.name)

        if not os.path.isdir(self.datadir):
            os.mkdir(self.datadir)

        self.h5name = os.path.join(self.datadir,'cache.hdf5')

        ## initialize equations and constants
        self.equations = self.init_equations()
        self.constants = self.init_constants()
        self.eqmss = self.calculate_eqmss()

        self.Nsystems = int(
            self.equations.shape[0]//self.Neqn_p_sys)
    
        ## deal with any tiling
        self.tileSystems()
        self.tileEquations()

        if dumpDebug:
            self.dumpToCDebugInput()

        for proto_file in ['ode_system.cu','ode_gold.c','device_dydt.cu']:
            self.make_ode_file(
                proto_file,
                self.Ntile,
                self.dconstants_string,
                self.jconstants_string)
            #self.make_RK2_file(self,self.Ntile)

    def tileSystems(self):
        self.equations = np.tile(self.equations,self.Nsystem_tile)
        self.constants = np.tile(self.constants,self.Nsystem_tile)
        self.eqmss = np.tile(self.eqmss,self.Nsystem_tile)
        self.Nsystems*=self.Nsystem_tile

    def tileEquations(self):
        self.orig_Neqn_p_sys = self.Neqn_p_sys
        self.Neqn_p_sys*=self.Ntile

        ## tile the ICs
        self.equations = np.concatenate(
            [np.tile(self.equations[
                i*self.Neqn_p_sys:
                (i+1)*self.Neqn_p_sys],
                self.Ntile)
            for i in range(self.Nsystems)])

        ## tile the eqmss
        if self.Ntile > 1:
            llist = []
            for i in range(self.Nsystems):
                foo = np.tile(
                    self.eqmss[
                        i*self.Neqn_p_sys//self.Ntile:
                        (i+1)*self.Neqn_p_sys//self.Ntile],self.Ntile)
                llist +=[foo]
            self.eqmss = np.concatenate(llist)

    def validate(self):
        self.init_constants()
        self.init_equations()
        self.calculate_jacobian()
        self.calculate_derivative()
        self.calculate_eqmss()
        self.dumpToODECache()

    def init_constants(self):
        raise NotImplementedError
        
    def init_equations(self):
        raise NotImplementedError

    def calculate_jacobian(self,jacobian_flat=None):
        if jacobian_flat is None:
            raise NotImplementedError
        else:
            tiled_jacobian_flat = np.zeros(self.Neqn_p_sys**2)
            ## tile the jacobian to make it block diagonal
            if self.Ntile > 1:
                for row_i in range(self.orig_Neqn_p_sys):
                    this_row = jacobian_flat[
                            self.orig_Neqn_p_sys*row_i:
                            self.orig_Neqn_p_sys*(row_i+1)]
                    for eqntile_i in range(self.Ntile):
                        ## how many rows should we skip?
                        offset = (eqntile_i*self.orig_Neqn_p_sys + row_i)
                        ## how many elements of new jacobian per row
                        offset*=self.orig_Neqn_p_sys*self.Ntile
                        ## how many columns to skip
                        offset+=self.orig_Neqn_p_sys*eqntile_i 

                        tiled_jacobian_flat[offset:offset+self.orig_Neqn_p_sys] = this_row 
                jacobian_flat = tiled_jacobian_flat
            ## indices above are in column major order to match cuBLAS specification
            return_val = jacobian_flat.astype(np.float32).reshape(self.Neqn_p_sys,self.Neqn_p_sys).T
            return return_val

    def calculate_derivative(self,rates=None):
        if rates is None:
            raise NotImplementedError
        else:
            ## tile the rates
            if self.Ntile > 1:
                rates = np.tile(rates,self.Ntile) 
            return rates.astype(np.float32)

    def calculate_eqmss(self):
        raise NotImplementedError

    def dumpToODECache(self,handle=None):
        if handle is not None:
            ## dump all the runtime metadata to the hdf5 file
            handle.attrs['Nsystems'] = self.Nsystems
            handle.attrs['Nequations_per_system'] = self.Neqn_p_sys
            handle.attrs['equation_labels'] = self.eqn_labels
            handle.attrs['absolute'] = self.absolute
            handle.attrs['relative'] = self.relative
            handle.attrs['Ntile'] = self.Ntile
            handle.attrs['Nsystem_tile'] = self.Nsystem_tile
            handle.attrs['n_integration_steps'] = self.n_integration_steps
            handle.attrs['n_output_steps'] = self.n_output_steps
        else:
            raise NotImplementedError


    derivative_suffix = "}\n"
    jacobian_suffix = "}\n"

### CUDA Solvers
    def runIntegratorOutput(
        self,
        integrator_fn,
        integrator_name,
        output_mode=None,
        print_flag = 0,
        python=False):

        equations = copy.copy(self.equations)
        constants = copy.copy(self.constants)

        ## initialize integration breakdown variables
        tcur = self.tnow
        dt = (self.tend-self.tnow)/self.n_output_steps
        equations_over_time = np.zeros((self.n_output_steps+1,len(equations)))
        nloops=0
        equations_over_time[nloops]=copy.deepcopy(equations)
        times = []
        times+=[tcur]
        nsteps = []
        walltimes = []

        while nloops < self.n_output_steps:#while tcur < tend:
            init_time = time.time()
            if not python:
                nsteps+=[
                    runCudaIntegrator(
                        integrator_fn,
                        tcur,tcur+dt,
                        self.n_integration_steps,
                        constants,equations,
                        self.Nsystems,self.Neqn_p_sys,
                        self.absolute,self.relative,
                        print_flag = print_flag)]
            else:
                systems_nsteps=0
                for system_i in range(self.Nsystems):
                    this_nsteps,this_equations = integrator_fn(
                            tcur,tcur+dt,
                            self.n_integration_steps,
                            constants[
                                self.nconst*system_i:
                                self.nconst*(system_i+1)],
                            equations[
                                self.Neqn_p_sys*system_i:
                                self.Neqn_p_sys*(system_i+1)],
                            self.calculate_derivative,
                            self.calculate_jacobian,
                            self.Neqn_p_sys,
                            self.absolute,self.relative)
                    systems_nsteps+=this_nsteps

                    equations[
                        self.Neqn_p_sys*system_i:
                        self.Neqn_p_sys*(system_i+1)] = this_equations

                nsteps+=[systems_nsteps]

            walltimes+=[time.time()-init_time]
            tcur+=dt
            times+=[tcur]
            nloops+=1
            equations_over_time[nloops]=copy.copy(equations)

        print('final (tcur=%.2f):'%tcur,np.round(equations_over_time.astype(float),5)[-1][:self.orig_Neqn_p_sys])

        if output_mode is not None:
        
            with h5py.File(self.h5name,output_mode) as handle:
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
        print('nsteps:',nsteps)
        print("total nsteps:",np.sum(nsteps))
        return np.array(times),np.array(equations_over_time)

    def dumpToCDebugInput(self):
        fname = os.path.join(
            self.datadir,
            self.name+'_debug.txt')
        #print("writing:",self.Nsystems,
            #"systems",self.Neqn_p_sys,
            #"equations per system to:",
            #fname)
        with open(fname,'w') as handle:
            
            handle.write(
                 "float tnow = %s;\n"%str(self.tnow))
            handle.write(
                "float tend = %s;\n" % str(self.tend/self.n_output_steps))
            handle.write(
                "int n_integration_steps = %s;\n" % str(self.n_integration_steps))
            
            fmt_equations = ["%.3e"%val if val != 0 else "0" for val in self.equations.flatten()]
            fmt_equations = ",".join(fmt_equations)
            fmt_equations = '{' + fmt_equations + '}'
            handle.write(
                "float equations[%d] = %s;\n" % (self.Nsystems*self.Neqn_p_sys,fmt_equations))
            handle.write(
                "float new_equations[%d] = %s;\n" % (self.Nsystems*self.Neqn_p_sys,fmt_equations))
            fmt_constants = ["%.3e"%val if val != 0 else "0" for val in self.constants.flatten()]
            fmt_constants = ",".join(fmt_constants)
            fmt_constants = '{' + fmt_constants + '}'
            handle.write(
                "float constants[%d] = %s;\n" % (self.Nsystems*self.nconst,fmt_constants))

            handle.write(
                "int Nsystems = %s;\n" % str(self.Nsystems))

            handle.write(
                "int Neqn_p_sys = %s;\n" % str(self.Neqn_p_sys))

            handle.write(
                "float absolute = %.2e;\n" % self.absolute)

            handle.write(
                "float relative = %.2e;\n" % self.relative)

def runCudaIntegrator(
    integrator,
    tnow,tend,
    n_integration_steps,
    constants,equations,
    Nsystems,Nequations_per_system,
    absolute,relative,
    print_flag=1):

    if print_flag:
        print("equations before:",equations)

    before = copy.copy(equations)
    nsteps =integrator(
        ctypes.c_float(np.float32(tnow)),
        ctypes.c_float(np.float32(tend)),
        ctypes.c_int(int(n_integration_steps)),
        constants.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        equations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(Nsystems),
        ctypes.c_int(Nequations_per_system),
        ctypes.c_float(absolute),
        ctypes.c_float(relative)
        )

    if print_flag:
        print("equations after %d steps:"%nsteps,equations.astype(np.float32))
        print(tnow,tend)
    return nsteps
