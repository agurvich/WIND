import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import h5py
from abg_python.all_utils import nameAxes
from distinct_colours import get_distinct
colors = get_distinct(5)
import pandas as pd


class ODECache(object):
    def __init__(self,fname,solvers = None):
        ## open the hdf5 file and bind them to this ODE system
        self.solvers = solvers
        self.open_ODE_cache(fname)
        print(self.solvers,'solvers used')
        
    def open_ODE_cache(self,fname):
        eqnsss = []
        nstepss = []
        timess = []
        labels=[]
        walltimess = []
        with h5py.File(fname,'r') as handle:
            for key in ['equation_labels','Nsystems','Nequations_per_system']:
                setattr(self,key,handle.attrs[key])

            self.eqmss = handle['Equilibrium']['eqmss'].value.reshape(
                -1,self.Nequations_per_system)

            self.grid_nHs = handle['Equilibrium/grid_nHs'].value
            self.grid_temperatures = handle['Equilibrium/grid_temperatures'].value
            self.grid_solar_metals = handle['Equilibrium/grid_solar_metals'].value

            for solver in handle.keys():
                if self.solvers is not None:
                    if solver not in self.solvers:
                        continue
                elif solver == 'Equilibrium':
                    continue
                labels+=[solver]
                solver = handle[solver]
                eqnsss+=[solver['equations_over_time'].value.reshape(
                   -1,self.Nsystems,self.Nequations_per_system).transpose(1,2,0)]

                times = solver['times'].value
                #timess+=[np.append([0],times[:-1])]
                timess+=[times]
                nstepss+=[solver['nsteps'].value]
                walltimess +=[solver['walltimes'].value]
                
        ## final shape is a solvers x Nsystems x eqns x timesteps
        self.equations_over_time = eqnsss
        self.nstepss = np.array(nstepss)
        self.timess = np.array(timess)
        self.solvers = labels
        self.walltimess = np.array(walltimess)
        
    def read_memory_usage(self,solver,size):
        memory_data = pd.read_csv("SIE_%s_memory.csv"%size)
        memory_data.timestamp=pd.to_datetime(memory_data.timestamp)
        memory_data.timestamp = memory_data.timestamp-memory_data.timestamp[0]
        xs = memory_data.timestamp.dt.total_seconds()
        ys = memory_data[' memory.used [MiB]']
        
        if 'memory_times' not in self.__dict__.keys():
            self.memory_times = [xs]
        else:
            self.memory_times.append(xs)
            
        if 'memory_usages' not in self.__dict__.keys():
            self.memory_usages = [ys]
        else:
            self.memory_usages.append(ys)
            
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
        nameAxes(ax,None,'t (s)','Global Memory Used MiB',logflag=(1,1),make_legend=True)
        if savefig is not None:
            fig.savefig(savefig)

    def plot_all_systems(self,savefig=None,**kwargs):
        plot_systems = min(self.Nsystems,100)
        grid_dim = int(np.ceil(np.sqrt(plot_systems)))
        fig,axs = plt.subplots(nrows=grid_dim,ncols=grid_dim)
        axs = axs.flatten()
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

        return fig
    
    def plot_system_over_time(
        self,
        system_index,
        ax=None,
        ylabel = '$n_X/n_\mathrm{H}$',
        xlabel = 't (yrs)',
        subtitle = None,
        plot_eqm = False,
        plot_legend_info = True,
        savefig = None):

        ax = plt.gca() if ax is None else ax
        fig = ax.get_figure()
        from matplotlib.lines import Line2D
        linestyles = ['--',':','-.','-']*2
        lws = [3,3,3,3]*2
        custom_lines = [Line2D([0], [0], color=colors[i],
            lw=lws[1],ls=linestyles[-1]) for i in range(len(self.solvers))]
        
        this_nstepss = []
        for solver_j,(this_solver_equations,times,nsteps,walltimes,label) in enumerate(
            zip(self.equations_over_time,self.timess,self.nstepss,self.walltimess,self.solvers)):
            for equation_i in range(len(this_solver_equations[system_index])):
                ys = this_solver_equations[system_index][equation_i]
                ax.plot(times,ys,c=colors[solver_j],ls = linestyles[equation_i],lw=lws[equation_i])
                
            this_nstepss +=[ np.sum(nsteps)]

        if plot_eqm:
            for equation_i in range(len(self.eqmss[system_index])):
                ax.axhline(
                self.eqmss[system_index][equation_i],
                color='k',#colors[equation_i],
                ls='-',xmin=.9,alpha=0.5)
                ax.text(times[-1]-equation_i,self.eqmss[system_index][equation_i],
                    self.equation_labels[equation_i].astype('U13'),
                    va='top',ha='left',
                    color='k',#colors[equation_i],
                    fontsize=14)
                
        if subtitle is None:
            subtitle = "log(nH) = %.1f - log(T) = %.1f - log(Z) = %.1f"%(
                self.grid_nHs[system_index],
                self.grid_temperatures[system_index],
                self.grid_solar_metals[system_index])

        nameAxes(ax,None,xlabel,ylabel,xlow=0,ylow=-0.1,
                 subtitle = subtitle,logflag=(0,0))
        
        walls = [np.sum(self.walltimess[solver_j])
            for solver_j in range(len(self.solvers))]

        nsystemss = [(self.Nsystems if solver != 'PY' else 1)
             for solver in self.solvers] 

        line_labels = ["%s - %d steps - %f s (%d systems)"%(solver,nsteps,wall,nsystems) 
             for solver,nsteps,wall,nsystems in zip(
                self.solvers,this_nstepss,walls,nsystemss)]

        if plot_legend_info:
            ax.legend(
                custom_lines, 
                line_labels,
                frameon = 0
                )

        fig.set_size_inches(16,9)
        if savefig is not None:
            fig.savefig(savefig)
