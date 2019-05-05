
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
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[20] = constants[2]*ne*this_block_state[21]
        -(constants[0]*ne + constants[1])*this_block_state[20]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[21] = -this_block_derivatives[20];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[22] = (constants[7]+constants[8])*ne*this_block_state[23] 
        - (constants[3]*ne+constants[4])*this_block_state[22];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[23] = constants[9]*ne*this_block_state[24] 
        + (constants[3]*ne+constants[4])*this_block_state[22]  
        - (constants[7]+constants[8])*ne*this_block_state[23] 
        - (constants[5]*ne+constants[6])*this_block_state[23];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[24] = (constants[5]*ne+constants[6])*this_block_state[23]
        -constants[9]*ne*this_block_state[24]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[25] = constants[2]*ne*this_block_state[26]
        -(constants[0]*ne + constants[1])*this_block_state[25]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[26] = -this_block_derivatives[25];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[27] = (constants[7]+constants[8])*ne*this_block_state[28] 
        - (constants[3]*ne+constants[4])*this_block_state[27];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[28] = constants[9]*ne*this_block_state[29] 
        + (constants[3]*ne+constants[4])*this_block_state[27]  
        - (constants[7]+constants[8])*ne*this_block_state[28] 
        - (constants[5]*ne+constants[6])*this_block_state[28];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[29] = (constants[5]*ne+constants[6])*this_block_state[28]
        -constants[9]*ne*this_block_state[29]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[30] = constants[2]*ne*this_block_state[31]
        -(constants[0]*ne + constants[1])*this_block_state[30]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[31] = -this_block_derivatives[30];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[32] = (constants[7]+constants[8])*ne*this_block_state[33] 
        - (constants[3]*ne+constants[4])*this_block_state[32];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[33] = constants[9]*ne*this_block_state[34] 
        + (constants[3]*ne+constants[4])*this_block_state[32]  
        - (constants[7]+constants[8])*ne*this_block_state[33] 
        - (constants[5]*ne+constants[6])*this_block_state[33];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[34] = (constants[5]*ne+constants[6])*this_block_state[33]
        -constants[9]*ne*this_block_state[34]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[35] = constants[2]*ne*this_block_state[36]
        -(constants[0]*ne + constants[1])*this_block_state[35]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[36] = -this_block_derivatives[35];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[37] = (constants[7]+constants[8])*ne*this_block_state[38] 
        - (constants[3]*ne+constants[4])*this_block_state[37];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[38] = constants[9]*ne*this_block_state[39] 
        + (constants[3]*ne+constants[4])*this_block_state[37]  
        - (constants[7]+constants[8])*ne*this_block_state[38] 
        - (constants[5]*ne+constants[6])*this_block_state[38];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[39] = (constants[5]*ne+constants[6])*this_block_state[38]
        -constants[9]*ne*this_block_state[39]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[40] = constants[2]*ne*this_block_state[41]
        -(constants[0]*ne + constants[1])*this_block_state[40]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[41] = -this_block_derivatives[40];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[42] = (constants[7]+constants[8])*ne*this_block_state[43] 
        - (constants[3]*ne+constants[4])*this_block_state[42];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[43] = constants[9]*ne*this_block_state[44] 
        + (constants[3]*ne+constants[4])*this_block_state[42]  
        - (constants[7]+constants[8])*ne*this_block_state[43] 
        - (constants[5]*ne+constants[6])*this_block_state[43];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[44] = (constants[5]*ne+constants[6])*this_block_state[43]
        -constants[9]*ne*this_block_state[44]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[45] = constants[2]*ne*this_block_state[46]
        -(constants[0]*ne + constants[1])*this_block_state[45]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[46] = -this_block_derivatives[45];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[47] = (constants[7]+constants[8])*ne*this_block_state[48] 
        - (constants[3]*ne+constants[4])*this_block_state[47];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[48] = constants[9]*ne*this_block_state[49] 
        + (constants[3]*ne+constants[4])*this_block_state[47]  
        - (constants[7]+constants[8])*ne*this_block_state[48] 
        - (constants[5]*ne+constants[6])*this_block_state[48];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[49] = (constants[5]*ne+constants[6])*this_block_state[48]
        -constants[9]*ne*this_block_state[49]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[50] = constants[2]*ne*this_block_state[51]
        -(constants[0]*ne + constants[1])*this_block_state[50]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[51] = -this_block_derivatives[50];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[52] = (constants[7]+constants[8])*ne*this_block_state[53] 
        - (constants[3]*ne+constants[4])*this_block_state[52];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[53] = constants[9]*ne*this_block_state[54] 
        + (constants[3]*ne+constants[4])*this_block_state[52]  
        - (constants[7]+constants[8])*ne*this_block_state[53] 
        - (constants[5]*ne+constants[6])*this_block_state[53];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[54] = (constants[5]*ne+constants[6])*this_block_state[53]
        -constants[9]*ne*this_block_state[54]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[55] = constants[2]*ne*this_block_state[56]
        -(constants[0]*ne + constants[1])*this_block_state[55]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[56] = -this_block_derivatives[55];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[57] = (constants[7]+constants[8])*ne*this_block_state[58] 
        - (constants[3]*ne+constants[4])*this_block_state[57];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[58] = constants[9]*ne*this_block_state[59] 
        + (constants[3]*ne+constants[4])*this_block_state[57]  
        - (constants[7]+constants[8])*ne*this_block_state[58] 
        - (constants[5]*ne+constants[6])*this_block_state[58];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[59] = (constants[5]*ne+constants[6])*this_block_state[58]
        -constants[9]*ne*this_block_state[59]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[60] = constants[2]*ne*this_block_state[61]
        -(constants[0]*ne + constants[1])*this_block_state[60]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[61] = -this_block_derivatives[60];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[62] = (constants[7]+constants[8])*ne*this_block_state[63] 
        - (constants[3]*ne+constants[4])*this_block_state[62];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[63] = constants[9]*ne*this_block_state[64] 
        + (constants[3]*ne+constants[4])*this_block_state[62]  
        - (constants[7]+constants[8])*ne*this_block_state[63] 
        - (constants[5]*ne+constants[6])*this_block_state[63];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[64] = (constants[5]*ne+constants[6])*this_block_state[63]
        -constants[9]*ne*this_block_state[64]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[65] = constants[2]*ne*this_block_state[66]
        -(constants[0]*ne + constants[1])*this_block_state[65]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[66] = -this_block_derivatives[65];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[67] = (constants[7]+constants[8])*ne*this_block_state[68] 
        - (constants[3]*ne+constants[4])*this_block_state[67];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[68] = constants[9]*ne*this_block_state[69] 
        + (constants[3]*ne+constants[4])*this_block_state[67]  
        - (constants[7]+constants[8])*ne*this_block_state[68] 
        - (constants[5]*ne+constants[6])*this_block_state[68];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[69] = (constants[5]*ne+constants[6])*this_block_state[68]
        -constants[9]*ne*this_block_state[69]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[70] = constants[2]*ne*this_block_state[71]
        -(constants[0]*ne + constants[1])*this_block_state[70]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[71] = -this_block_derivatives[70];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[72] = (constants[7]+constants[8])*ne*this_block_state[73] 
        - (constants[3]*ne+constants[4])*this_block_state[72];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[73] = constants[9]*ne*this_block_state[74] 
        + (constants[3]*ne+constants[4])*this_block_state[72]  
        - (constants[7]+constants[8])*ne*this_block_state[73] 
        - (constants[5]*ne+constants[6])*this_block_state[73];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[74] = (constants[5]*ne+constants[6])*this_block_state[73]
        -constants[9]*ne*this_block_state[74]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[75] = constants[2]*ne*this_block_state[76]
        -(constants[0]*ne + constants[1])*this_block_state[75]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[76] = -this_block_derivatives[75];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[77] = (constants[7]+constants[8])*ne*this_block_state[78] 
        - (constants[3]*ne+constants[4])*this_block_state[77];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[78] = constants[9]*ne*this_block_state[79] 
        + (constants[3]*ne+constants[4])*this_block_state[77]  
        - (constants[7]+constants[8])*ne*this_block_state[78] 
        - (constants[5]*ne+constants[6])*this_block_state[78];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[79] = (constants[5]*ne+constants[6])*this_block_state[78]
        -constants[9]*ne*this_block_state[79]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[80] = constants[2]*ne*this_block_state[81]
        -(constants[0]*ne + constants[1])*this_block_state[80]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[81] = -this_block_derivatives[80];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[82] = (constants[7]+constants[8])*ne*this_block_state[83] 
        - (constants[3]*ne+constants[4])*this_block_state[82];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[83] = constants[9]*ne*this_block_state[84] 
        + (constants[3]*ne+constants[4])*this_block_state[82]  
        - (constants[7]+constants[8])*ne*this_block_state[83] 
        - (constants[5]*ne+constants[6])*this_block_state[83];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[84] = (constants[5]*ne+constants[6])*this_block_state[83]
        -constants[9]*ne*this_block_state[84]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[85] = constants[2]*ne*this_block_state[86]
        -(constants[0]*ne + constants[1])*this_block_state[85]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[86] = -this_block_derivatives[85];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[87] = (constants[7]+constants[8])*ne*this_block_state[88] 
        - (constants[3]*ne+constants[4])*this_block_state[87];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[88] = constants[9]*ne*this_block_state[89] 
        + (constants[3]*ne+constants[4])*this_block_state[87]  
        - (constants[7]+constants[8])*ne*this_block_state[88] 
        - (constants[5]*ne+constants[6])*this_block_state[88];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[89] = (constants[5]*ne+constants[6])*this_block_state[88]
        -constants[9]*ne*this_block_state[89]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[90] = constants[2]*ne*this_block_state[91]
        -(constants[0]*ne + constants[1])*this_block_state[90]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[91] = -this_block_derivatives[90];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[92] = (constants[7]+constants[8])*ne*this_block_state[93] 
        - (constants[3]*ne+constants[4])*this_block_state[92];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[93] = constants[9]*ne*this_block_state[94] 
        + (constants[3]*ne+constants[4])*this_block_state[92]  
        - (constants[7]+constants[8])*ne*this_block_state[93] 
        - (constants[5]*ne+constants[6])*this_block_state[93];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[94] = (constants[5]*ne+constants[6])*this_block_state[93]
        -constants[9]*ne*this_block_state[94]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[95] = constants[2]*ne*this_block_state[96]
        -(constants[0]*ne + constants[1])*this_block_state[95]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[96] = -this_block_derivatives[95];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[97] = (constants[7]+constants[8])*ne*this_block_state[98] 
        - (constants[3]*ne+constants[4])*this_block_state[97];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[98] = constants[9]*ne*this_block_state[99] 
        + (constants[3]*ne+constants[4])*this_block_state[97]  
        - (constants[7]+constants[8])*ne*this_block_state[98] 
        - (constants[5]*ne+constants[6])*this_block_state[98];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[99] = (constants[5]*ne+constants[6])*this_block_state[98]
        -constants[9]*ne*this_block_state[99]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[100] = constants[2]*ne*this_block_state[101]
        -(constants[0]*ne + constants[1])*this_block_state[100]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[101] = -this_block_derivatives[100];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[102] = (constants[7]+constants[8])*ne*this_block_state[103] 
        - (constants[3]*ne+constants[4])*this_block_state[102];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[103] = constants[9]*ne*this_block_state[104] 
        + (constants[3]*ne+constants[4])*this_block_state[102]  
        - (constants[7]+constants[8])*ne*this_block_state[103] 
        - (constants[5]*ne+constants[6])*this_block_state[103];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[104] = (constants[5]*ne+constants[6])*this_block_state[103]
        -constants[9]*ne*this_block_state[104]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[105] = constants[2]*ne*this_block_state[106]
        -(constants[0]*ne + constants[1])*this_block_state[105]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[106] = -this_block_derivatives[105];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[107] = (constants[7]+constants[8])*ne*this_block_state[108] 
        - (constants[3]*ne+constants[4])*this_block_state[107];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[108] = constants[9]*ne*this_block_state[109] 
        + (constants[3]*ne+constants[4])*this_block_state[107]  
        - (constants[7]+constants[8])*ne*this_block_state[108] 
        - (constants[5]*ne+constants[6])*this_block_state[108];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[109] = (constants[5]*ne+constants[6])*this_block_state[108]
        -constants[9]*ne*this_block_state[109]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[110] = constants[2]*ne*this_block_state[111]
        -(constants[0]*ne + constants[1])*this_block_state[110]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[111] = -this_block_derivatives[110];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[112] = (constants[7]+constants[8])*ne*this_block_state[113] 
        - (constants[3]*ne+constants[4])*this_block_state[112];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[113] = constants[9]*ne*this_block_state[114] 
        + (constants[3]*ne+constants[4])*this_block_state[112]  
        - (constants[7]+constants[8])*ne*this_block_state[113] 
        - (constants[5]*ne+constants[6])*this_block_state[113];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[114] = (constants[5]*ne+constants[6])*this_block_state[113]
        -constants[9]*ne*this_block_state[114]; 
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
    this_block_jacobian[116] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[115] = -this_block_jacobian[116]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[232] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[233] = this_block_jacobian[232]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[349] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[347] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[348] = -this_block_jacobian[347] - 
        this_block_jacobian[349]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[464] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[463] = -this_block_jacobian[464];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[580] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[581] = -this_block_jacobian[580]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[696] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[695] = -this_block_jacobian[696]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[812] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[813] = this_block_jacobian[812]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[929] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[927] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[928] = -this_block_jacobian[927] - 
        this_block_jacobian[929]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[1044] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[1043] = -this_block_jacobian[1044];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[1160] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[1161] = -this_block_jacobian[1160]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[1276] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[1275] = -this_block_jacobian[1276]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[1392] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[1393] = this_block_jacobian[1392]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[1509] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[1507] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[1508] = -this_block_jacobian[1507] - 
        this_block_jacobian[1509]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[1624] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[1623] = -this_block_jacobian[1624];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[1740] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[1741] = -this_block_jacobian[1740]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[1856] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[1855] = -this_block_jacobian[1856]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[1972] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[1973] = this_block_jacobian[1972]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[2089] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[2087] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[2088] = -this_block_jacobian[2087] - 
        this_block_jacobian[2089]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[2204] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[2203] = -this_block_jacobian[2204];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[2320] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[2321] = -this_block_jacobian[2320]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[2436] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[2435] = -this_block_jacobian[2436]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[2552] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[2553] = this_block_jacobian[2552]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[2669] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[2667] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[2668] = -this_block_jacobian[2667] - 
        this_block_jacobian[2669]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[2784] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[2783] = -this_block_jacobian[2784];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[2900] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[2901] = -this_block_jacobian[2900]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[3016] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[3015] = -this_block_jacobian[3016]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[3132] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[3133] = this_block_jacobian[3132]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[3249] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[3247] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[3248] = -this_block_jacobian[3247] - 
        this_block_jacobian[3249]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[3364] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[3363] = -this_block_jacobian[3364];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[3480] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[3481] = -this_block_jacobian[3480]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[3596] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[3595] = -this_block_jacobian[3596]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[3712] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[3713] = this_block_jacobian[3712]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[3829] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[3827] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[3828] = -this_block_jacobian[3827] - 
        this_block_jacobian[3829]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[3944] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[3943] = -this_block_jacobian[3944];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[4060] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[4061] = -this_block_jacobian[4060]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[4176] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[4175] = -this_block_jacobian[4176]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[4292] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[4293] = this_block_jacobian[4292]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[4409] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[4407] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[4408] = -this_block_jacobian[4407] - 
        this_block_jacobian[4409]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[4524] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[4523] = -this_block_jacobian[4524];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[4640] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[4641] = -this_block_jacobian[4640]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[4756] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[4755] = -this_block_jacobian[4756]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[4872] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[4873] = this_block_jacobian[4872]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[4989] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[4987] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[4988] = -this_block_jacobian[4987] - 
        this_block_jacobian[4989]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[5104] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[5103] = -this_block_jacobian[5104];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[5220] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[5221] = -this_block_jacobian[5220]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[5336] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[5335] = -this_block_jacobian[5336]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[5452] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[5453] = this_block_jacobian[5452]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[5569] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[5567] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[5568] = -this_block_jacobian[5567] - 
        this_block_jacobian[5569]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[5684] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[5683] = -this_block_jacobian[5684];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[5800] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[5801] = -this_block_jacobian[5800]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[5916] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[5915] = -this_block_jacobian[5916]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[6032] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[6033] = this_block_jacobian[6032]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[6149] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[6147] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[6148] = -this_block_jacobian[6147] - 
        this_block_jacobian[6149]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[6264] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[6263] = -this_block_jacobian[6264];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[6380] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[6381] = -this_block_jacobian[6380]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[6496] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[6495] = -this_block_jacobian[6496]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[6612] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[6613] = this_block_jacobian[6612]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[6729] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[6727] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[6728] = -this_block_jacobian[6727] - 
        this_block_jacobian[6729]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[6844] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[6843] = -this_block_jacobian[6844];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[6960] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[6961] = -this_block_jacobian[6960]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[7076] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[7075] = -this_block_jacobian[7076]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[7192] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[7193] = this_block_jacobian[7192]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[7309] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[7307] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[7308] = -this_block_jacobian[7307] - 
        this_block_jacobian[7309]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[7424] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[7423] = -this_block_jacobian[7424];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[7540] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[7541] = -this_block_jacobian[7540]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[7656] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[7655] = -this_block_jacobian[7656]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[7772] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[7773] = this_block_jacobian[7772]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[7889] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[7887] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[7888] = -this_block_jacobian[7887] - 
        this_block_jacobian[7889]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[8004] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[8003] = -this_block_jacobian[8004];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[8120] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[8121] = -this_block_jacobian[8120]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[8236] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[8235] = -this_block_jacobian[8236]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[8352] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[8353] = this_block_jacobian[8352]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[8469] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[8467] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[8468] = -this_block_jacobian[8467] - 
        this_block_jacobian[8469]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[8584] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[8583] = -this_block_jacobian[8584];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[8700] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[8701] = -this_block_jacobian[8700]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[8816] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[8815] = -this_block_jacobian[8816]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[8932] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[8933] = this_block_jacobian[8932]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[9049] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[9047] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[9048] = -this_block_jacobian[9047] - 
        this_block_jacobian[9049]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[9164] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[9163] = -this_block_jacobian[9164];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[9280] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[9281] = -this_block_jacobian[9280]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[9396] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[9395] = -this_block_jacobian[9396]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[9512] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[9513] = this_block_jacobian[9512]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[9629] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[9627] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[9628] = -this_block_jacobian[9627] - 
        this_block_jacobian[9629]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[9744] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[9743] = -this_block_jacobian[9744];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[9860] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[9861] = -this_block_jacobian[9860]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[9976] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[9975] = -this_block_jacobian[9976]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[10092] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[10093] = this_block_jacobian[10092]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[10209] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[10207] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[10208] = -this_block_jacobian[10207] - 
        this_block_jacobian[10209]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[10324] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[10323] = -this_block_jacobian[10324];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[10440] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[10441] = -this_block_jacobian[10440]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[10556] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[10555] = -this_block_jacobian[10556]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[10672] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[10673] = this_block_jacobian[10672]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[10789] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[10787] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[10788] = -this_block_jacobian[10787] - 
        this_block_jacobian[10789]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[10904] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[10903] = -this_block_jacobian[10904];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[11020] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[11021] = -this_block_jacobian[11020]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[11136] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[11135] = -this_block_jacobian[11136]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[11252] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[11253] = this_block_jacobian[11252]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[11369] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[11367] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[11368] = -this_block_jacobian[11367] - 
        this_block_jacobian[11369]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[11484] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[11483] = -this_block_jacobian[11484];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[11600] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[11601] = -this_block_jacobian[11600]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[11716] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[11715] = -this_block_jacobian[11716]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[11832] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[11833] = this_block_jacobian[11832]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[11949] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[11947] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[11948] = -this_block_jacobian[11947] - 
        this_block_jacobian[11949]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[12064] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[12063] = -this_block_jacobian[12064];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[12180] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[12181] = -this_block_jacobian[12180]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[12296] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[12295] = -this_block_jacobian[12296]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[12412] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[12413] = this_block_jacobian[12412]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[12529] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[12527] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[12528] = -this_block_jacobian[12527] - 
        this_block_jacobian[12529]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[12644] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[12643] = -this_block_jacobian[12644];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[12760] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[12761] = -this_block_jacobian[12760]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[12876] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[12875] = -this_block_jacobian[12876]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[12992] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[12993] = this_block_jacobian[12992]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[13109] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[13107] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[13108] = -this_block_jacobian[13107] - 
        this_block_jacobian[13109]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[13224] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[13223] = -this_block_jacobian[13224];//He+ : 9-alpha_(He++)ne
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
