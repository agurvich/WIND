from ode_systems.katz96 import Katz96
from ode_systems.NR_test import NR_test
import sys
import getopt


def main(
    Ntile=2,
    system_name='katz'):
    if system_name == 'katz':
        system = Katz96(Ntile=Ntile)
    elif system_name == 'NR':
        system = NR_test(Ntile=Ntile)
    
    ## make the ode.cu file
    system.preprocess()
    
    
if __name__ == '__main__':
    argv = sys.argv[1:]
    opts,args = getopt.getopt(argv,'',[
        'system_name=','Ntile='])

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
