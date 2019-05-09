import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import h5py
from abg_python.all_utils import nameAxes
from distinct_colours import get_distinct
colors = get_distinct(5)
import pandas as pd
import os



class ODECache(object):
    def gb_neqns(self):
        if 'Katz96' in self.name:
            return 5
        elif 'NR_test' in self.name:
            return 2
        else:
            raise KeyError("What system is this?")

    def __repr__(self):
        strr = "ODECache %s"%self.name
        for solver in self.solvers:
            strr+=' - %s'%solver
        return strr

    def __init__(
            self,
            name,
            fname,
            solvers = None,
            datadir=None,
            color = None):

        self.name=name
        self.datadir = '../data' if datadir is None else datadir
        self.datadir = os.path.join(self.datadir,self.name)
        ## open the hdf5 file and bind them to this ODE system
        self.solvers = solvers
        self.open_ODE_cache(fname)

        self.memory_times = {}
        self.memory_usages ={}
        
        self.color = get_distinct(0) if color is None else color

        for solver in self.solvers:
            try:
                self.read_memory_usage(solver)
            except OSError:
                print("No memory profiling information for",
                    self,solver)

        print(self.solvers,'solvers used')
        
    def open_ODE_cache(self,fname):
        eqnsss = {}
        nstepss = []
        timess = {}
        labels=[]
        walltimess = []
        print(self.datadir,fname)
        with h5py.File(
            os.path.join(
            self.datadir,
            fname),'r') as handle:
            try:
                for key in [
                    'equation_labels',
                    'Nsystems',
                    'Nequations_per_system']:
                    setattr(self,key,handle.attrs[key])
            except:
                print(handle.keys())
                raise

            self.eqmss = handle['Equilibrium']['eqmss'].value.reshape(
                -1,self.Nequations_per_system)

            ## unpack any configuration information that 
            ##  might've been saved
            for key in handle['Equilibrium'].keys():
                if key == 'eqmss':
                    continue
                setattr(self,key,handle['Equilibrium'][key].value) 

            for solver in handle.keys():
                if self.solvers is not None:
                    if solver not in self.solvers:
                        continue
                elif solver == 'Equilibrium':
                    continue
                labels+=[solver]
                solver = handle[solver]
                eqnsss[labels[-1]]=(
                    solver['equations_over_time'].value.reshape(
                   -1,self.Nsystems,
                    self.Nequations_per_system).transpose(1,2,0))

                times = solver['times'].value
                #timess+=[np.append([0],times[:-1])]
                timess[labels[-1]]=times

                nstepss+=[solver['nsteps'].value]
                walltimess +=[solver['walltimes'].value]
                
        ## final shape is a solvers x Nsystems x eqns x timesteps
        self.equations_over_time = eqnsss

        self.timess = timess

        self.solvers = labels

        self.walltimess = dict(zip(self.solvers,
            np.array(walltimess)))
        
        self.nstepss = dict(zip(self.solvers,
            np.array(nstepss)))

    def read_memory_usage(self,solver):
        memory_data = pd.read_csv(
            os.path.join(
                self.datadir,
                "%s_%s_memory.csv"%(self.name,solver)))
        memory_data.timestamp=pd.to_datetime(
            memory_data.timestamp)
        memory_data.timestamp = (memory_data.timestamp-
            memory_data.timestamp[0])
        xs = memory_data.timestamp.dt.total_seconds().values
        ys = memory_data[' memory.used [MiB]'].values
        
        self.memory_times[solver] = xs
        self.memory_usages[solver] = ys
            
    def check_speedup(self):
        pass
        #walls = np.mean(walltimess/nstepss,axis=1)

        #print("%s is "%labels[1],walls[0]/walls[1], "times faster than %s per step"%labels[0])
        #print("%s is "%labels[1],walls[2]/walls[1], "times faster than %s per step"%labels[2])

    ##### PLOTTING FUNCTIONS
    def plot_memory_usage_versus_size(self,sizes,ax=None,savefig=None):
        ax = plt.gca() if ax is None else ax
        fig = ax.get_figure()
        for size in sizes:
            self.read_memory_usage('SIE',size)

            ax.plot(self.memory_times[-1],self.memory_usages[-1],'o',
                    markersize=2,markeredgewidth=2,label=size)
        nameAxes(
            ax,None,
            't (s)','Global Memory Used MiB',
            logflag=(1,1),make_legend=True)

        if savefig is not None:
            fig.savefig(savefig)

    def plot_all_systems(self,savefig=None,
        axs=None,**kwargs):
        plot_systems = min(self.Nsystems,49)
        grid_dim = int(np.ceil(np.sqrt(plot_systems)))
        if axs is None:
            fig,axs = plt.subplots(nrows=grid_dim,ncols=grid_dim)
            if self.Nsystems > 1:
                axs = axs.flatten()
            else:
                axs = [axs]
        else:
            fig = axs[0].get_figure()
            plot_systems = len(axs)

        for system_index in range(plot_systems):
            ax = axs[system_index]
            self.plot_system_over_time(
                system_index,
                ax,
                **kwargs)
                

        scale = grid_dim/4.0
        fig.set_size_inches(32*scale,32*scale)

        if savefig is not None:
            fig.savefig(savefig)

        return fig,axs
    
    def plot_system_over_time(
        self,
        system_index,
        ax=None,
        subtitle = None,
        plot_eqm = False,
        plot_legend_info = True,
        savefig = None,
        use_color = None,
        solvers = None,
        **kwargs):

        ax = plt.gca() if ax is None else ax
        fig = ax.get_figure()
        from matplotlib.lines import Line2D
        linestyles = ['--',':','-.','-']
        lws = [4,4,4,4]*2

        custom_lines = [Line2D(
            [0], [0], color=colors[i],
            lw=lws[1],ls=linestyles[-1]) 
            for i in range(len(self.solvers))]
        
        solvers = self.solvers if solvers is None else solvers

        this_total_nstepss = []
        ## for each solver that was used (e.g. RK2, SIE, etc...)
        for solver_j,solver in enumerate(solvers):
            this_solver_equations = self.equations_over_time[solver]
            times = self.timess[solver]
            nsteps = self.nstepss[solver]
            walltimes =self.walltimess[solver]

            base_neqn = self.gb_neqns()
            yss = this_solver_equations[system_index]
            if base_neqn != self.Nequations_per_system:
                new_yss = [] 
                minss = []
                maxss = []
                for base_index in range(base_neqn):
                    new_yss += [np.median(
                        yss[base_index::base_neqn],
                        axis = 0)]

                    minss += [np.min(
                        yss[base_index::base_neqn],
                        axis = 0)]

                    maxss += [np.max(
                        yss[base_index::base_neqn],
                        axis = 0)]

                minss = np.array(minss)
                maxss = np.array(maxss)
                yss = new_yss
            else:
                minss = maxss = None

            ## plot each equation in the system
            for equation_i in range(len(yss)):

                color = (colors[solver_j] if use_color is 
                    None else use_color)

                ys = yss[equation_i]


                ax.plot(
                    times,ys,
                    c=color,
                    ls = linestyles[equation_i%len(linestyles)],
                    lw=lws[equation_i%len(lws)])

                if minss is not None:
                    ax.fill_between(
                        times,
                        minss[equation_i],
                        maxss[equation_i],
                        color=color,
                        alpha = 1)
                
            ## save the total number of steps so we can put 
            ##  it in the legend...
            total_steps = np.sum(nsteps)
            if solver == 'RK2':
                total_steps/=self.Nsystems
            this_total_nstepss +=[ total_steps]

        ## plot converged value
        if plot_eqm:
            for equation_i in range(base_neqn):
                ax.axhline(
                    self.eqmss[system_index][equation_i],
                    color='w',#colors[equation_i],
                    ls='-',xmin=.9,
                    lw=2.5,alpha=0.5)
                ax.text(
                    times[-1],
                    self.eqmss[system_index][equation_i],
                    (self.equation_labels[
                        equation_i%len(self.equation_labels)]
                        .astype('U13')),
                    va='bottom',
                    ha='right' if equation_i<4 else 'left',
                    #colors[equation_i],
                    fontsize=14)
                
        if subtitle is None:
            try:
                subtitle = ("log(nH) = %.1f"%
                    self.grid_nHs[system_index]+
                    " - log(T) = %.1f"% 
                    self.grid_temperatures[system_index])
                    #+ " - log(Z) = %.1f"%
                    #self.grid_solar_metals[system_index])
            except AttributeError:
                pass

        nameAxes(
            ax,None,
            subtitle = subtitle,
            logflag=(0,0),
            **kwargs)
        
        walls = [np.sum(self.walltimess[solver])
            for solver in self.solvers]

        nsystemss = [(self.Nsystems if solver != 'PY' else 1)
             for solver in self.solvers] 

        line_labels = [
            "     %s - %d steps \n %f s (%d systems)"%(
            solver,nsteps*nsystems,wall,nsystems) 
            for solver,nsteps,wall,nsystems in zip(
                self.solvers,this_total_nstepss,walls,nsystemss)]

        loc = 0 if 'loc' not in kwargs else kwargs['loc']
        if plot_legend_info:
            ax.legend(
                custom_lines, 
                line_labels,
                frameon = 0,
                loc = loc
                )

        fig.set_size_inches(16,9)
        if savefig is not None:
            fig.savefig(savefig)
        

def get_fname(name):
    split_name = name.split('_')
    return name#'_'.join(split_name[:1]+split_name[-1:])

class MultiODECache(object):
    def __getitem__(self,index):
        return self.ode_caches[index]

    def __init__(
        self,
        names,
        datadir,
        solvers = None):
        ## open the hdf5 file and bind them to this ODE system
        self.solvers = solvers
        self.ode_caches = []
        fnames = [get_fname(name)+'.hdf5' for name in names]
        self.colors = np.tile(get_distinct(min(len(names),12)),2)
        for name,fname,color in zip(names,fnames,self.colors):
            self.ode_caches+=[
                ODECache(
                    name,
                    fname,
                    solvers,
                    datadir=datadir,
                    color = color)]

        print(self.ode_caches)


        if 'Katz96' in names[0]:
            self.system_name = 'Katz96'
        else:
            self.system_name = 'NR_test'
        #self.system_name = '_'.join(names[0].split('_')[:-1])
        
        params = np.array([name[len(self.system_name)+1:].split('_')
             for name in names])
        
        self.ntiles = params[:,-1].astype(int)
        if params.shape[1]>1:
            self.params = params[:,:-1]
