import sys
import numpy as np
import os


cuda_dir = os.path.realpath(__file__)
for i in range(4):
    cuda_dir = os.path.split(cuda_dir)[0]
cuda_dir = os.path.join(cuda_dir,'cuda')

print("WIND cuda directory:",cuda_dir)

def reindex(index,Ntile,Ndim,this_tile):
    nrow = index // Ndim
    return index + (Ntile-1)*Ndim*nrow + (Ndim*Ndim*Ntile+Ndim)*this_tile

def make_string(loop_fn,Ntile,prefix,suffix):
    strr = prefix
    for this_tile in range(Ntile):
        strr+=loop_fn(this_tile,Ntile)
    return strr + suffix

def make_ode_file(system,Ntile):
    strr = ""
    with open(
        os.path.join(
            cuda_dir,
            'ode_prefix.cu'),
        'r') as handle:

        for line in handle.readlines():
            strr+=line

    strr+=make_string(
        system.make_derivative_block,
        Ntile,
        system.derivative_prefix,
        system.derivative_suffix)

    strr+=make_string(
        system.make_jacobian_block,
        Ntile,
        system.jacobian_prefix,
        system.jacobian_suffix)

    with open(
        os.path.join(
            cuda_dir,
            'ode_suffix.cu'),
        'r') as handle:

        for line in handle.readlines():
            strr+=line

    with open(
        os.path.join(
            system.datadir,
            '%s_preprocess_ode.cu'%system.name,
            ),'w') as handle:

        handle.write(strr)

    return make_ode_file

def make_RK2_file(system,Ntile):
    strr = ''
    strr += make_string(
        system.make_RK2_block,
        Ntile,
        system.RK2_prefix,
        system.RK2_suffix)


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
            system.datadir,
            '%s_preprocess_RK2_kernel.cu'%
            system.name),'w') as handle:
        handle.write(strr)
