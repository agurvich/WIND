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
    equations+=eqn_offset;
    constants+=bid*NUM_CONST;
    float * dydt = d_derivatives_flat+eqn_offset;

    // constraint equation, ne = nH+ + nHe+ + 2*nHe++
    float ne = equations[1]+equations[3]+equations[4]*2.0;

    /* constants = [
        0-Gamma_(e,H0), 1-Gamma_(gamma,H0), 
        2-alpha_(H+),
        3-Gamma_(e,He0), 4-Gamma_(gamma,He0), 
        5-Gamma_(e,He+), 6-Gamma_(gamma,He+),
        7-alpha_(He+),
        8-alpha_(d),
        9-alpha_(He++)
        ] 
    */
    // H0 : alpha_(H+) ne nH+ - (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0
    dydt[0] = constants[2]*ne*equations[1]
            -(constants[0]*ne + constants[1])*equations[0]; 
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    dydt[1] = -constants[2]*ne*equations[1]
            +(constants[0]*ne + constants[1])*equations[0]; 
    // He0 :(alpha_(He+)+alpha_(d)) ne nHe+ - (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
    dydt[2] = (constants[7]+constants[8])*ne*equations[3] 
        - (constants[3]*ne+constants[4])*equations[2];
    // He+ : 
    //  alpha_(He++) ne nHe++ 
    //  + (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
    //  - (alpha_(He+)+alpha_(d)) ne nHe+ 
    //  - (Gamma_(e,He+)ne + Gamma_(gamma,He+)) nHe+
    dydt[3] = constants[9]*ne*equations[4] 
        + (constants[3]*ne+constants[4])*equations[2]  
        - (constants[7]+constants[8])*ne*equations[3] 
        - (constants[5]*ne+constants[6])*equations[3];
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    dydt[4] = (constants[5]*ne+constants[6])*equations[3]
        -constants[9]*ne*equations[4]; 
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
    equations+=eqn_offset;
    constants+=bid*NUM_CONST;

    float * Jacobian = d_Jacobianss[bid];

    // constraint equation, ne = nH+ + nHe+ + 2*nHe++
    float ne = equations[1]+equations[3]+equations[4]*2.0;

    /* constants = [
        0-Gamma_(e,H0), 1-Gamma_(gamma,H0), 
        2-alpha_(H+),
        3-Gamma_(e,He0), 4-Gamma_(gamma,He0), 
        5-Gamma_(e,He+), 6-Gamma_(gamma,He+),
        7-alpha_(He+),
        8-alpha_(d),
        9-alpha_(He++)
        ] 
    */

   
    // H0
    Jacobian[0] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    Jacobian[1] = -Jacobian[0]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    Jacobian[6] = -constants[2]*ne; // H+ -alpha_(H+)ne
    Jacobian[5] = -Jacobian[6]; // H0 : 2-alpha_(H+)ne
        
    // He0
    Jacobian[12] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    Jacobian[13] = Jacobian[12]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    Jacobian[19] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    Jacobian[17] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    Jacobian[18] = -Jacobian[17] - Jacobian[19]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    Jacobian[24] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    Jacobian[23] = -Jacobian[24];//He+ : 9-alpha_(He++)ne
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