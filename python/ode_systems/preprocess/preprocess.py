import sys
import numpy as np

def reindex(index,Ntile,Ndim,this_tile):
    nrow = index // Ndim
    return index + (Ntile-1)*Ndim*nrow + (Ndim*Ndim*Ntile+Ndim)*this_tile

def make_jacobian_string(system,Ntile,prefix,suffix):
    strr = prefix
    for this_tile in range(Ntile):
        strr+=system.make_jacobian_block(this_tile,Ntile)
    return strr + suffix

def make_derivative_string(system,Ntile,prefix,suffix):
    strr = prefix
    for this_tile in range(Ntile):
        strr+=system.get_derivative_block(this_tile,Ntile)
    return strr + suffix

def make_ode_file(system,Ntile):
    strr = file_prefix

    strr+=make_derivative_string(
        system,
        Ntile,
        system.derivative_prefix,
        system.derivative_suffix)

    strr+=make_jacobian_string(
        system,
        Ntile,
        system.jacobian_prefix,
        system.jacobian_suffix)

    strr+=file_suffix

    with open('precompile_cu_files/%s_preprocess_ode.cu'%system.name,'w') as handle:
        handle.write(strr)

    return make_ode_file

file_suffix = """
    void resetSystem(
        float ** d_derivatives,
        float * d_derivatives_flat,
        float ** d_Jacobianss,
        float * d_Jacobianss_flat,
        float * d_constants,
        float * d_current_state_flat,
        float * jacobian_zeros,
        int Nsystems,
        int Neqn_p_sys,
        float tnow){

        dim3 ode_gridDim;
        configureGrid(
            Nsystems,Neqn_p_sys,
            NULL,
            NULL,
            &ode_gridDim,
            NULL);


        // evaluate the derivative function at tnow
        calculateDerivatives<<<ode_gridDim,1>>>(
            d_derivatives_flat,
            d_constants,
            d_current_state_flat,
            Nsystems,
            Neqn_p_sys,
            tnow);

        // reset the jacobian, which has been replaced by (I-hJ)^-1
        cudaMemcpy(
            d_Jacobianss_flat,jacobian_zeros,
            Nsystems*Neqn_p_sys*Neqn_p_sys*sizeof(float),
            cudaMemcpyHostToDevice);

        calculateJacobians<<<ode_gridDim,1>>>(
            d_Jacobianss,
            d_constants,
            d_current_state_flat,
            Nsystems,
            Neqn_p_sys,
            tnow);
    }
    """

file_prefix ="""#include "ode.h"
    #include "vector_kernels.h"
    #include <stdio.h>

    /* ---------------- CUDA Thread Block Organization ------------ */
    void configureGrid(
        int Nsystems,int Neqn_p_sys,
        int * p_threads_per_block,
        dim3 * p_matrix_gridDim,
        dim3 * p_ode_gridDim,
        dim3 * p_vector_gridDim){

        int threads_per_block = min(Neqn_p_sys,MAX_THREADS_PER_BLOCK);
        int x_blocks_per_grid = 1+Neqn_p_sys/MAX_THREADS_PER_BLOCK;
        int y_blocks_per_grid = min(Nsystems,MAX_BLOCKS_PER_GRID);
        int z_blocks_per_grid = 1+Nsystems/MAX_BLOCKS_PER_GRID;

        dim3 matrix_gridDim(
            x_blocks_per_grid*Neqn_p_sys,
            y_blocks_per_grid,
            z_blocks_per_grid);

        dim3 ode_gridDim(
            1,
            y_blocks_per_grid,
            z_blocks_per_grid);

        dim3 vector_gridDim(
                x_blocks_per_grid,
                y_blocks_per_grid,
                z_blocks_per_grid);

        if (p_threads_per_block != NULL){
            *p_threads_per_block = threads_per_block;
        }

        if (p_matrix_gridDim != NULL){
            *p_matrix_gridDim = matrix_gridDim;
        }

        if (p_ode_gridDim != NULL){
            *p_ode_gridDim = ode_gridDim;
        }

        if (p_vector_gridDim != NULL){
            *p_vector_gridDim = vector_gridDim;
        }
    }


    /* ------------------------------------------------------------ */

    __device__ int get_system_index(){
        return blockIdx.z*gridDim.y + blockIdx.y; 
    }
"""
