
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

    // constraint equation, ne = nH+ + nHe+ + 2*nHe++
    float ne = equations[eqn_offset+1]+equations[eqn_offset+3]+equations[eqn_offset+4]*2.0;

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
    // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[0] = constants[2]*ne*this_block_state[1]
        -(constants[0]*ne + constants[1])*this_block_state[0]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[1] = -this_block_derivatives[0];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[2] = (constants[7]+constants[8])*ne*this_block_state[3] 
        - (constants[3]*ne+constants[4])*this_block_state[2];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[3] = constants[9]*ne*this_block_state[4] 
        + (constants[3]*ne+constants[4])*this_block_state[2]  
        - (constants[7]+constants[8])*ne*this_block_state[3] 
        - (constants[5]*ne+constants[6])*this_block_state[3];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[4] = (constants[5]*ne+constants[6])*this_block_state[3]
        -constants[9]*ne*this_block_state[4]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[5] = constants[2]*ne*this_block_state[6]
        -(constants[0]*ne + constants[1])*this_block_state[5]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[6] = -this_block_derivatives[5];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[7] = (constants[7]+constants[8])*ne*this_block_state[8] 
        - (constants[3]*ne+constants[4])*this_block_state[7];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[8] = constants[9]*ne*this_block_state[9] 
        + (constants[3]*ne+constants[4])*this_block_state[7]  
        - (constants[7]+constants[8])*ne*this_block_state[8] 
        - (constants[5]*ne+constants[6])*this_block_state[8];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[9] = (constants[5]*ne+constants[6])*this_block_state[8]
        -constants[9]*ne*this_block_state[9]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[10] = constants[2]*ne*this_block_state[11]
        -(constants[0]*ne + constants[1])*this_block_state[10]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[11] = -this_block_derivatives[10];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[12] = (constants[7]+constants[8])*ne*this_block_state[13] 
        - (constants[3]*ne+constants[4])*this_block_state[12];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[13] = constants[9]*ne*this_block_state[14] 
        + (constants[3]*ne+constants[4])*this_block_state[12]  
        - (constants[7]+constants[8])*ne*this_block_state[13] 
        - (constants[5]*ne+constants[6])*this_block_state[13];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[14] = (constants[5]*ne+constants[6])*this_block_state[13]
        -constants[9]*ne*this_block_state[14]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[15] = constants[2]*ne*this_block_state[16]
        -(constants[0]*ne + constants[1])*this_block_state[15]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[16] = -this_block_derivatives[15];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[17] = (constants[7]+constants[8])*ne*this_block_state[18] 
        - (constants[3]*ne+constants[4])*this_block_state[17];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[18] = constants[9]*ne*this_block_state[19] 
        + (constants[3]*ne+constants[4])*this_block_state[17]  
        - (constants[7]+constants[8])*ne*this_block_state[18] 
        - (constants[5]*ne+constants[6])*this_block_state[18];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[19] = (constants[5]*ne+constants[6])*this_block_state[18]
        -constants[9]*ne*this_block_state[19]; 
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

    // constraint equation, ne = nH+ + nHe+ + 2*nHe++
    float ne = this_block_state[1]+this_block_state[3]+this_block_state[4]*2.0;

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
    this_block_jacobian[0] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[1] = -this_block_jacobian[0]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[21] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[20] = -this_block_jacobian[21]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[42] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[43] = this_block_jacobian[42]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[64] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[62] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[63] = -this_block_jacobian[62] - 
        this_block_jacobian[64]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[84] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[83] = -this_block_jacobian[84];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[105] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[106] = -this_block_jacobian[105]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[126] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[125] = -this_block_jacobian[126]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[147] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[148] = this_block_jacobian[147]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[169] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[167] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[168] = -this_block_jacobian[167] - 
        this_block_jacobian[169]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[189] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[188] = -this_block_jacobian[189];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[210] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[211] = -this_block_jacobian[210]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[231] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[230] = -this_block_jacobian[231]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[252] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[253] = this_block_jacobian[252]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[274] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[272] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[273] = -this_block_jacobian[272] - 
        this_block_jacobian[274]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[294] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[293] = -this_block_jacobian[294];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[315] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[316] = -this_block_jacobian[315]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[336] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[335] = -this_block_jacobian[336]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[357] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[358] = this_block_jacobian[357]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[379] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[377] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[378] = -this_block_jacobian[377] - 
        this_block_jacobian[379]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[399] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[398] = -this_block_jacobian[399];//He+ : 9-alpha_(He++)ne
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
