from ode_systems.katz96 import Katz96 as k96_system
from ode_systems.NR_test import NR_test as nr_test_system

import sys
import getopt


def main(
    Ntile=2,
    Nsystem_tile=1,
    nsteps=1,
    system_name='Katz96'):

    if system_name == 'Katz96':
        system = k96_system(
            Ntile=Ntile,
            Nsystem_tile=Nsystem_tile,
            nsteps=nsteps)
    elif system_name == 'NR_test':
        system = nr_test_system(
            Ntile=Ntile,
            Nsystem_tile=Nsystem_tile,
            nsteps=nsteps)
    else:
        raise ValueError("pick Katz96 or NR_test")

    ## make the ode.cu file
    system.preprocess()
    
if __name__ == '__main__':
    argv = sys.argv[1:]
    opts,args = getopt.getopt(argv,'',[
        'system_name=','Ntile=','Nsystem_tile=','nsteps='])

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
    print(dict(opts),'keywords passed')
    main(**dict(opts))
