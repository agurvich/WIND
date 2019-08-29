import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import h5py
from abg_python.all_utils import nameAxes,bufferAxesLabels,fitLeastSq,add_curve_label
from abg_python.distinct_colours import get_distinct
from abg_python.snapshot_utils import inv_chimes_dict
import pandas as pd
import os


all_colors = get_distinct(10)

arch_colors = all_colors[6:]

plot_labels = {
    'SIE':'SIE-gpu',
    'SIEgold':'SIE-cpu',
    'RK2':'RK2-gpu',
    'RK2gold':'RK2-cpu',
    'CHIMES':'CVODE'
}

linestyles = {
    'SIE':'-',
    'SIEgold':'--',
    'RK2':'-',
    'RK2gold':'--',
    'CHIMES':'--'
}




colors = {
    'SIE':all_colors[0],
    'SIEgold':all_colors[1],
    'RK2':all_colors[2],
    'RK2gold':all_colors[3],
    'CHIMES':all_colors[4]
}

class ODECache(object):
    def gb_neqns(self):
        if 'Katz96' in self.name:
            return 5
        elif 'NR_test' in self.name:
            return 2
        elif 'FullChimes' in self.name:
            return 10
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

        ## read the tolerances from the path... not the best idea
        self.datadir = os.path.join(self.datadir,self.name)
        ## open the hdf5 file and bind them to this ODE system
        self.solvers = solvers
        self.open_ODE_cache(fname)

        self.memory_times = {}
        self.memory_usages ={}
        
        self.color = get_distinct(0) if color is None else color

        for solver in self.solvers:
            try:
                if 'gold' in solver:
                    self.read_memory_usage_gold(solver) 
                else:
                    self.read_memory_usage(solver)
            except OSError:
                print("No memory profiling information for",
                    self,solver)

        ##print(self.solvers,'solvers used')
        
    def open_ODE_cache(self,fname):
        eqnsss = {}
        nstepss = []
        timess = {}
        labels=[]
        walltimess = []
        print(self.datadir)
        with h5py.File(
            os.path.join(
            self.datadir,
            fname),'r') as handle:
            try:
                for key in [
                    'Nequations_per_system',
                    'Nsystem_tile',
                     'Nsystems',
                     'Ntile',
                     'absolute',
                     'equation_labels',
                     'n_integration_steps',
                     'n_output_steps',
                     'relative']:
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
                print(solver['equations_over_time'].value.shape)
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

    def read_memory_usage_gold(self,solver):
        memfile = os.path.join(
            self.datadir,
            "%s_memory.dat"%(solver))

        memory_data = pd.read_csv(
            memfile,
            delimiter=' ',
            skiprows=1,
            usecols=[1,2],
            names=['MEM','timestamp'])

        memory_data.timestamp=pd.to_datetime(
                    memory_data.timestamp,unit='s')

        memory_data.timestamp = (memory_data.timestamp-
            memory_data.timestamp[0])

        xs = memory_data.timestamp.dt.total_seconds().values
        ys = memory_data.MEM.values

        self.memory_times[solver] = xs
        self.memory_usages[solver] = ys

    def read_memory_usage(self,solver):
        memory_data = pd.read_csv(
            os.path.join(
                self.datadir,
                "%s_memory.csv"%(solver)))
        memory_data.timestamp=pd.to_datetime(
            memory_data.timestamp)
        memory_data.timestamp = (memory_data.timestamp-
            memory_data.timestamp[0])
        xs = memory_data.timestamp.dt.total_seconds().values
        ## convert to MB
        ys = memory_data[' memory.used [MiB]'].values/1024**2*1e6
        
        self.memory_times[solver] = xs
        self.memory_usages[solver] = ys
            
    def check_speedup(self):
        pass
        #walls = np.mean(walltimess/nstepss,axis=1)

        #print("%s is "%labels[1],walls[0]/walls[1], "times faster than %s per step"%labels[0])
        #print("%s is "%labels[1],walls[2]/walls[1], "times faster than %s per step"%labels[2])

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
        make_legend_info = True,
        savefig = None,
        use_color = None,
        solvers = None,
        equation_indices = None,
        annotate=True,
        **kwargs):

        ax = plt.gca() if ax is None else ax
        fig = ax.get_figure()
        from matplotlib.lines import Line2D
        linestyles = ['--',':','-.','-']
        lws = [4,4*0.8,4*0.8**2,0.5]*2

        custom_lines = [Line2D(
            [0], [0], color=colors[self.solvers[i]],
            lw=lws[1],ls=linestyles[-1]) 
            for i in range(len(self.solvers))]
        
        solvers = self.solvers if solvers is None else solvers
        equation_indices = range(200) if equation_indices is None else equation_indices

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

                if equation_i not in equation_indices:
                    continue

                color = (colors[self.solvers[solver_j]] if use_color is 
                    None else use_color)

                ys = yss[equation_i]

                ax.plot(
                    times,ys,
                    c=color,
                    ls = linestyles[equation_i%len(linestyles)],
                    lw = lws[solver_j],
                    label=inv_chimes_dict[equation_i])

                if minss is not None:
                    ax.fill_between(
                        times,
                        minss[equation_i],
                        maxss[equation_i],
                        color=color,
                        alpha = 1)

                if annotate and solver_j==0:
                    ax.text(
                        times[-1],
                        ys[-1],
                        inv_chimes_dict[equation_i],
                        va='bottom',
                        ha='right' if equation_i<4 else 'left',
                        #colors[equation_i],
                        fontsize=14)
                
            ## save the total number of steps so we can put 
            ##  it in the legend...
            total_steps = np.sum(nsteps)
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
            make_legend=1,
            **kwargs)
        
        walls = [np.sum(self.walltimess[solver])
            for solver in self.solvers]

        nsystemss = [(self.Nsystems if solver != 'PY' else 1)
             for solver in self.solvers] 

        line_labels = [
            "     %s - %s steps \n %f s (%d systems)"%(
            plot_labels[solver],
            "%d"%nsteps if nsteps>0 else "?",
            wall,nsystems) 
            for solver,nsteps,wall,nsystems in zip(
                self.solvers,this_total_nstepss,walls,nsystemss)]

        loc = 0 if 'loc' not in kwargs else kwargs['loc']
        if make_legend_info:
            if not annotate:
                legend = ax.legend(frameon=False,loc=2)
                ax.add_artist(legend)

            legend1 = ax.legend(
                custom_lines, 
                line_labels,
                frameon = 0,
                loc = loc
                )

            ax.add_artist(legend1)

        fig.set_size_inches(16,9)
        if savefig is not None:
            fig.savefig(savefig)
        
    def plot_nsteps_histogram(self,ax,solver,label=None):
        xs = self.timess[solver]
        ys = self.nstepss[solver]/self.Nsystems
        label = self.name if label is None else label

        ax.step(
            xs[1:],ys,
            lw=3,
            label=label,
            where='post',
            color=self.color)

        nameAxes(
            ax,None,
            '$t_\mathrm{ode}$ (yrs)',
            '$N_\mathrm{steps}$',
            logflag=(0,1))

        return ax

class MultiODECache(object):
    def __getitem__(self,index):
        return self.ode_caches[index]

    def __repr__(self):
        strr = "%s with %d caches"%(self.name,len(self.ode_caches))
        return strr

    def __init__(
        self,
        name,
        names,
        datadir,
        solvers = None):
        self.name = name
        ## open the hdf5 file and bind them to this ODE system
        self.solvers = solvers
        self.ode_caches = []
        fnames = ['cache.hdf5' for name in names]
        self.colors = np.tile(get_distinct(min(len(names),12)),2)
        
        self.datadir = datadir
        ## add support for tolerances
        for name,fname,color in zip(names,fnames,self.colors):
            self.ode_caches+=[
                ODECache(
                    name,
                    fname,
                    solvers,
                    datadir=datadir,
                    color = color)]

        if 'Katz96' in names[0]:
            self.system_name = 'Katz96'
        else:
            self.system_name = 'NR_test'
        
        for key in [
            'Nequations_per_system',
            'Nsystem_tile',
             'Nsystems',
             'Ntile',
             'absolute',
             'equation_labels',
             'n_integration_steps',
             'n_output_steps',
             'relative']:
            vals = []
            for cache in self.ode_caches:
                vals+=[getattr(cache,key)]
            setattr(self,key+'s',vals)

    def plot_tts_vs_xs(
        self,
        ax=None,
        x_function=None,
        y_function=None,
        eqn_legend=True,
        do_fit=True,
        pos_rels=None,
        vas = None,
        slopes = None,
        units = None,
        var_label = None,
        bestfit_key = None,
        **kwargs):
        
        var_label = 'eqn' if var_label is None else var_label
        slopes = [1,1,1,1] if slopes is None else slopes
        units = 's' if units is None else units
        xss,yss,eqn_labels = [],[],[]
        pos_rels = np.repeat(0.9,len(self[0].solvers)) if pos_rels is None else pos_rels
        bestfit_key = 'best_tts' if bestfit_key is None else bestfit_key
        ## what is the x value of each ode_cache?
        if x_function is None:
            raise ValueError("Provide an x-lambda function")

        if y_function is None:
            y_function = lambda x,solver: np.sum(x.walltimess[solver])

        vas = ['top']*len(self[0].solvers) if vas is None else vas

        if ax is None:
            ## create a new figure
            fig = plt.figure()
            ax = plt.gca()
        else:
            fig = ax.get_figure()

        ## accumulator arrays to hold best fit
        test_xss = []
        test_yss = []
        bestfits = []
        ## loop through each solver
        for solver_i,solver in enumerate([
            'RK2','RK2gold','SIE','SIEgold']):
            ## generate y values
            try:
                ys = np.array([
                    y_function(ode_cache,solver) for ode_cache 
                    in self.ode_caches])
                xs = np.array([
                    x_function(ode_cache) for ode_cache 
                    in self.ode_caches])
            except:
                continue
            
            if do_fit:
        
                slope = slopes[solver_i]

                b = ys[0]
                p0=[np.mean(ys)]
                fn = lambda pars,xs: pars[0]*xs**slope+b
                pars=fitLeastSq(fn,p0,xs,ys)
                ## label the line
                slope_label = "$^%d$"%slope * (slope >1)
                a = pars[0]
                ## figure out what is the best set of units to use
                bestfits+=[(solver,b,a)]

                b_scale = -np.floor(np.log10(b))
                if b_scale >= 4:
                    b_metric_units = "$\mu$"
                    b_scale=6
                elif b_scale >= 1:
                    b_metric_units = "m"
                    b_scale=3
                else:
                    b_scale = 0
                    b_metric_units=""

                a_scale = -np.floor(np.log10(a))
                if a_scale >= 4:
                    a_metric_units = "$\mu$"
                    a_scale=6
                elif a_scale >= 1:
                    a_metric_units = "m"
                    a_scale=3
                else:
                    a_scale = 0
                    a_metric_units=""

                if units[0] == 'M' and a_metric_units != "":

                    if a_metric_units == '$\mu$':
                        a_metric_units = ""
                    elif a_metric_units == 'm':
                        a_metric_units = "K"
                    unit_str = a_metric_units+units[1:]
                else:
                    unit_str = a_metric_units + units

                b_scale=10**b_scale
                a_scale=10**a_scale
        
                eqn_label = r'%d %s + %.1f %s%s'%(
                    b*b_scale,
                    b_metric_units+units,
                    a*a_scale,
                    unit_str, 
                    '/'*(slope > 0)+var_label+slope_label)

                eqn_labels+=[eqn_label]
                    
                test_xs = 10**np.linspace(
                    np.log10(np.min(xs)),
                    np.log10(np.max(xs)),100)
                test_ys = fn(pars,test_xs)
    
                test_xss+=[test_xs]
                test_yss+=[test_ys]
                ## plot the best fit line
                ax.plot(
                    test_xs,test_ys,
                    c=colors[solver],
                    label=plot_labels[solver],
                    lw=3,ls=linestyles[solver])

                ax.plot(
                    xs[:len(ys)],ys,
                    '.',markersize=16,
                    lw=3,c=colors[solver])

            else:
                ax.plot(xs[:len(ys)],ys,lw=3,
                    label=plot_labels[solver],c=colors[solver])

        ## make the legend
        nameAxes(
            ax,None,#xname,yname,
            logflag=(1,1),
            make_legend=True,
            **kwargs) 

        if not eqn_legend:
            for solver_i,(test_xs,test_ys,eqn_label) in enumerate(
                zip(test_xss,test_yss,eqn_labels)):
                add_curve_label(
                    ax,
                    test_xs,test_ys,
                    eqn_label,
                    label_pos_rel = pos_rels[solver_i],
                    color = colors[self[0].solvers[solver_i]],
                    va=vas[solver_i],
                    fontsize=14,
                    weight = 'bold')
        fig.set_size_inches(8,4.5) 
        

        keys,vals0,vals1 = zip(*bestfits)
        keys = list(keys)
        vals0 = list(vals0)
        vals1 = list(vals1)
        ## TODO remove this because this failed only on BW
        ##  and I'm rerunning
        if 'RK2gold' not in keys:
            keys=keys[:1]+['RK2gold']+keys[1:]
            vals0=vals0[:1]+[np.nan]+vals0[1:]
            vals1=vals1[:1]+[np.nan]+vals1[1:]
            eqn_labels = eqn_labels[:1]+['']+eqn_labels[1:]

        if 'SIEgold' not in keys:
            keys+=['SIEgold']
            vals0+=[np.nan]
            vals1+=[np.nan]
            eqn_labels += ['']
            
        bestdict = dict(zip(keys,zip(vals0,vals1,slopes,eqn_labels)))
        setattr(self,bestfit_key,bestdict) 

        return fig,ax
        

    def plot_memory_usages(
        self,
        savefig=False,
        legend_ax_i=1,
        label_fn = None):

        ## handle default label fn
        label_fn = (lambda x: x.name ) if label_fn is None else label_fn

        ## create a new figure
        fig,axs = plt.subplots(
            ncols=2,
            nrows=2,
            sharex=True,
            sharey=True)
        axs = axs.flatten()

        ## plot each solver onto its axis
        for ax_i,(ax,solver) in enumerate(zip(
            axs,['RK2','RK2gold','SIE','SIEgold'])):
            for ode_i,ode_cache in enumerate(self):
                try:
                    xs = ode_cache.memory_times[solver]
                    ys = ode_cache.memory_usages[solver]
                except KeyError:
                    continue
                neqn = ode_cache.Nequations_per_system
                label = label_fn(ode_cache)
                ax.plot(xs,ys,'.',c=ode_cache.color,label=label)
        
            nameAxes(
                ax,None,'t (s)', 'memory (MB)',
                logflag=(1,1),
                ylow=80,yhigh=3000,
                supertitle=solver,
                subtitle="%s - Nsystems: %d"%(
                    self.system_name,
                    ode_cache.Nsystems),
                slackify=True,
                make_legend=ax_i==legend_ax_i,loc=0)
        
        ## squish the axes together
        fig.set_size_inches(8,8)
        plt.subplots_adjust(hspace=0,wspace=0)

        bufferAxesLabels(axs,2,2,
                ylabels=True,share_ylabel='Memory (MB)',
                xlabels=True,share_xlabel='t (s)',
                label_offset=0.025)

        ## save if necessary
        if savefig:
            fig.set_facecolor('k')
            fig.savefig('%s_memory_vs_time.pdf'%self.system_name,
                facecolor='k')


    def plot_all_nsteps_histogram(self,label_fn = None,savefig=False):
        fig,axs = plt.subplots(ncols=2,nrows=2,sharex=True,sharey=True)
        axs = axs.flatten()

        label_fn = (lambda x: x.name) if label_fn is None else label_fn
        ## plot each solver's steps
        for ax,solver in zip(axs,['RK2','RK2gold','SIE','SIEgold']):
            for ode_cache in self:
                ode_cache.plot_nsteps_histogram(
                    ax,solver,label = label_fn(ode_cache))
            nameAxes(ax,None,None,None,
                subtitle=solver)

        ## squish axes together
        bufferAxesLabels(axs,2,2,
            ylabels=True,
            share_xlabel='$t_\mathrm{ode}$ (yrs)',
            xlabels=True,
            share_ylabel= '$N_\mathrm{steps}/N_\mathrm{systems}$',
            label_offset=0.05)
        plt.subplots_adjust(hspace=0,wspace=0)
        fig.set_size_inches(12,6.75)

        if savefig:
            fig.savefig(
                "%s_nsteps_per_noutputsteps.pdf"%
                    multi_odes.system_name,
                facecolor='k')

        return fig,axs

    def readDeviceQuery(self,loud=False):
        gpufile = os.path.join(
            self.datadir,"gpu.txt")

        ## baby helper function
        def clean_line(line):
            split = np.array(line.split(' '))
            split = split[split!='']
            return split

        try:
            with open(gpufile,'r') as handle:
                for line in handle.readlines():
                    ## read the global memory off
                    if 'Total amount of global memory' in line:
                        split=clean_line(line)
                        index = np.where(split=='MBytes')[0][0]
                        self.glob_memory = int(split[index-1])

                        if loud:
                            print('Total glob memory:',
                                self.total_glob_memory)
                        
                    ## read off the number of cuda cores
                    elif 'CUDA Cores' in line:
                        split=clean_line(line)
                        index = np.where(split=='CUDA')[0][1]
                        self.ncores = int(split[index-1])
                        if loud:
                            print("Total cuda cores:",self.ncores) 
                    
                    ## read off the GPU clock rate
                    elif 'GPU Max Clock rate' in line:
                        split=clean_line(line)
                        index = np.where(split=='MHz')[0][0]
                        self.clock_rate = int(split[index-1])
                        if loud:
                            print("Clock rate in MHz:",
                                self.clock_rate) 
        except IOError:
            print("Can't find %s"%gpufile)

class MultiArch(object):
    def __getitem__(self,index):
        return self.archs[index]

    def __repr__(self):
        return str(self.archs)

    def __init__(self,archs):
        self.archs = archs

    def plot_scaling(
        self,
        axs=None,
        scaling=None,
        **kwargs):

        if axs is None:
            fig,axs = plt.subplots(2,2,sharex=True,sharey=True)
            axs = axs.flatten()
        else:
            fig = axs[0].get_figure()

        scaling = 'best_tts' if scaling is None else scaling
        fn = lambda b,a,slope,xs: b + a*xs**slope
        test_xs = 10**np.linspace(
            np.log10(1),np.log10(1000),25)

        for solver,ax in zip(
            getattr(self[-1],scaling).keys(),
            axs):

            try:
                bestfit_dict = getattr(self.archs[0],scaling)
                b,a,slope,eqn_label = bestfit_dict[solver] 
                baseline_ys = fn(b,a,slope,test_xs)
            except:
                continue

            for arch_i,arch in enumerate(self.archs[1:]):
                bestfit_dict = getattr(arch,scaling)


                ##unpack the values
                try:
                    b,a,slope,eqn_label = bestfit_dict[solver] 
                except:
                    continue

                ax.plot(
                    test_xs,
                    baseline_ys/fn(b,a,slope,test_xs),
                    color = arch_colors[arch_i],
                    label=arch.name,
                    ##ls = linestyles[solver],
                    lw = 2.5)

            nameAxes(
                ax,
                None,
                logflag=(1,1),
                subtitle=solver,
                make_legend=1,
                **kwargs)

        plt.subplots_adjust(hspace=0,wspace=0)
        bufferAxesLabels(
            axs,
            2,2,
            ylabels=True,
            share_ylabel='Speedup vs. %s'%self.archs[0].name,
            xlabels=True,
            share_xlabel = 'Neqn_p_sys')
        fig.set_size_inches(8,8)








