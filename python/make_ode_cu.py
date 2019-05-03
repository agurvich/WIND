from ode_systems.katz96 import Katz96
import sys

if __name__ == '__main__':
    arg = sys.argv[1:]
    if len(arg):
        arg = int(arg[0])
    else:
        arg = 1

    system = Katz96(0,1,1)
    system.preprocess(arg)
