
#include "ode.h"
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
__global__ void calculateDerivatives(
    float * d_derivatives_flat, 
    float * constants, 
    float * equations,
    int Nsystems,
    int Neqn_p_sys,
    float time){
    // isolate this system 

    int bid = get_system_index();
    // don't need to do anything, no system corresponds to this thread-block
    if (bid >= Nsystems){
        return;
    }

    int eqn_offset = bid*Neqn_p_sys;
    float * this_block_state = equations+eqn_offset;
    float * this_block_derivatives = d_derivatives_flat+eqn_offset;

    // eq. 16.6.1 in NR 
    this_block_derivatives[0] = 998.0*this_block_state[0] + 1998. * this_block_state[1];
    this_block_derivatives[1] = -999.0*this_block_state[0] - 1999.0*this_block_state[1];

    // eq. 16.6.1 in NR 
    this_block_derivatives[2] = 998.0*this_block_state[2] + 1998. * this_block_state[3];
    this_block_derivatives[3] = -999.0*this_block_state[2] - 1999.0*this_block_state[3];

    // eq. 16.6.1 in NR 
    this_block_derivatives[4] = 998.0*this_block_state[4] + 1998. * this_block_state[5];
    this_block_derivatives[5] = -999.0*this_block_state[4] - 1999.0*this_block_state[5];

    // eq. 16.6.1 in NR 
    this_block_derivatives[6] = 998.0*this_block_state[6] + 1998. * this_block_state[7];
    this_block_derivatives[7] = -999.0*this_block_state[6] - 1999.0*this_block_state[7];

    // eq. 16.6.1 in NR 
    this_block_derivatives[8] = 998.0*this_block_state[8] + 1998. * this_block_state[9];
    this_block_derivatives[9] = -999.0*this_block_state[8] - 1999.0*this_block_state[9];
}
__global__ void calculateJacobians(
    float **d_Jacobianss, 
    float * constants,
    float * equations,
    int Nsystems,
    int Neqn_p_sys,
    float time){

    // isolate this system 
    int bid = get_system_index();

    // don't need to do anything, no system corresponds to this thread-block
    if (bid >= Nsystems){
        return;
    }

    int eqn_offset = bid*Neqn_p_sys;
    float * this_block_state = equations+eqn_offset;
    float * this_block_jacobian = d_Jacobianss[bid];

    this_block_jacobian[0] = 998.0;
    this_block_jacobian[1] = -999.0;
    this_block_jacobian[10] = 1998.0;
    this_block_jacobian[11] = -1999.0;

    this_block_jacobian[22] = 998.0;
    this_block_jacobian[23] = -999.0;
    this_block_jacobian[32] = 1998.0;
    this_block_jacobian[33] = -1999.0;

    this_block_jacobian[44] = 998.0;
    this_block_jacobian[45] = -999.0;
    this_block_jacobian[54] = 1998.0;
    this_block_jacobian[55] = -1999.0;

    this_block_jacobian[66] = 998.0;
    this_block_jacobian[67] = -999.0;
    this_block_jacobian[76] = 1998.0;
    this_block_jacobian[77] = -1999.0;

    this_block_jacobian[88] = 998.0;
    this_block_jacobian[89] = -999.0;
    this_block_jacobian[98] = 1998.0;
    this_block_jacobian[99] = -1999.0;
}

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


    if (d_derivatives_flat !=NULL){
        // evaluate the derivative function at tnow
        calculateDerivatives<<<ode_gridDim,1>>>(
            d_derivatives_flat,
            d_constants,
            d_current_state_flat,
            Nsystems,
            Neqn_p_sys,
            tnow);
    }

    if (d_Jacobianss_flat != NULL){
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
}
