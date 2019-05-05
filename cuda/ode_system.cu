
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
    this_block_jacobian[111] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[110] = -this_block_jacobian[111]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[222] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[223] = this_block_jacobian[222]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[334] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[332] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[333] = -this_block_jacobian[332] - 
        this_block_jacobian[334]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[444] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[443] = -this_block_jacobian[444];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[555] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[556] = -this_block_jacobian[555]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[666] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[665] = -this_block_jacobian[666]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[777] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[778] = this_block_jacobian[777]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[889] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[887] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[888] = -this_block_jacobian[887] - 
        this_block_jacobian[889]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[999] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[998] = -this_block_jacobian[999];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[1110] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[1111] = -this_block_jacobian[1110]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[1221] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[1220] = -this_block_jacobian[1221]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[1332] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[1333] = this_block_jacobian[1332]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[1444] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[1442] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[1443] = -this_block_jacobian[1442] - 
        this_block_jacobian[1444]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[1554] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[1553] = -this_block_jacobian[1554];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[1665] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[1666] = -this_block_jacobian[1665]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[1776] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[1775] = -this_block_jacobian[1776]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[1887] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[1888] = this_block_jacobian[1887]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[1999] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[1997] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[1998] = -this_block_jacobian[1997] - 
        this_block_jacobian[1999]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[2109] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[2108] = -this_block_jacobian[2109];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[2220] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[2221] = -this_block_jacobian[2220]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[2331] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[2330] = -this_block_jacobian[2331]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[2442] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[2443] = this_block_jacobian[2442]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[2554] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[2552] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[2553] = -this_block_jacobian[2552] - 
        this_block_jacobian[2554]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[2664] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[2663] = -this_block_jacobian[2664];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[2775] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[2776] = -this_block_jacobian[2775]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[2886] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[2885] = -this_block_jacobian[2886]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[2997] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[2998] = this_block_jacobian[2997]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[3109] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[3107] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[3108] = -this_block_jacobian[3107] - 
        this_block_jacobian[3109]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[3219] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[3218] = -this_block_jacobian[3219];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[3330] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[3331] = -this_block_jacobian[3330]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[3441] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[3440] = -this_block_jacobian[3441]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[3552] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[3553] = this_block_jacobian[3552]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[3664] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[3662] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[3663] = -this_block_jacobian[3662] - 
        this_block_jacobian[3664]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[3774] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[3773] = -this_block_jacobian[3774];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[3885] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[3886] = -this_block_jacobian[3885]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[3996] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[3995] = -this_block_jacobian[3996]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[4107] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[4108] = this_block_jacobian[4107]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[4219] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[4217] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[4218] = -this_block_jacobian[4217] - 
        this_block_jacobian[4219]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[4329] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[4328] = -this_block_jacobian[4329];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[4440] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[4441] = -this_block_jacobian[4440]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[4551] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[4550] = -this_block_jacobian[4551]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[4662] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[4663] = this_block_jacobian[4662]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[4774] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[4772] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[4773] = -this_block_jacobian[4772] - 
        this_block_jacobian[4774]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[4884] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[4883] = -this_block_jacobian[4884];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[4995] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[4996] = -this_block_jacobian[4995]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[5106] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[5105] = -this_block_jacobian[5106]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[5217] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[5218] = this_block_jacobian[5217]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[5329] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[5327] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[5328] = -this_block_jacobian[5327] - 
        this_block_jacobian[5329]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[5439] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[5438] = -this_block_jacobian[5439];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[5550] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[5551] = -this_block_jacobian[5550]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[5661] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[5660] = -this_block_jacobian[5661]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[5772] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[5773] = this_block_jacobian[5772]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[5884] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[5882] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[5883] = -this_block_jacobian[5882] - 
        this_block_jacobian[5884]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[5994] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[5993] = -this_block_jacobian[5994];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[6105] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[6106] = -this_block_jacobian[6105]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[6216] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[6215] = -this_block_jacobian[6216]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[6327] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[6328] = this_block_jacobian[6327]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[6439] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[6437] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[6438] = -this_block_jacobian[6437] - 
        this_block_jacobian[6439]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[6549] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[6548] = -this_block_jacobian[6549];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[6660] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[6661] = -this_block_jacobian[6660]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[6771] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[6770] = -this_block_jacobian[6771]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[6882] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[6883] = this_block_jacobian[6882]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[6994] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[6992] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[6993] = -this_block_jacobian[6992] - 
        this_block_jacobian[6994]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[7104] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[7103] = -this_block_jacobian[7104];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[7215] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[7216] = -this_block_jacobian[7215]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[7326] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[7325] = -this_block_jacobian[7326]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[7437] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[7438] = this_block_jacobian[7437]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[7549] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[7547] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[7548] = -this_block_jacobian[7547] - 
        this_block_jacobian[7549]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[7659] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[7658] = -this_block_jacobian[7659];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[7770] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[7771] = -this_block_jacobian[7770]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[7881] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[7880] = -this_block_jacobian[7881]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[7992] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[7993] = this_block_jacobian[7992]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[8104] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[8102] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[8103] = -this_block_jacobian[8102] - 
        this_block_jacobian[8104]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[8214] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[8213] = -this_block_jacobian[8214];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[8325] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[8326] = -this_block_jacobian[8325]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[8436] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[8435] = -this_block_jacobian[8436]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[8547] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[8548] = this_block_jacobian[8547]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[8659] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[8657] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[8658] = -this_block_jacobian[8657] - 
        this_block_jacobian[8659]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[8769] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[8768] = -this_block_jacobian[8769];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[8880] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[8881] = -this_block_jacobian[8880]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[8991] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[8990] = -this_block_jacobian[8991]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[9102] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[9103] = this_block_jacobian[9102]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[9214] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[9212] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[9213] = -this_block_jacobian[9212] - 
        this_block_jacobian[9214]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[9324] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[9323] = -this_block_jacobian[9324];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[9435] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[9436] = -this_block_jacobian[9435]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[9546] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[9545] = -this_block_jacobian[9546]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[9657] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[9658] = this_block_jacobian[9657]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[9769] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[9767] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[9768] = -this_block_jacobian[9767] - 
        this_block_jacobian[9769]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[9879] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[9878] = -this_block_jacobian[9879];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[9990] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[9991] = -this_block_jacobian[9990]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[10101] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[10100] = -this_block_jacobian[10101]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[10212] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[10213] = this_block_jacobian[10212]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[10324] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[10322] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[10323] = -this_block_jacobian[10322] - 
        this_block_jacobian[10324]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[10434] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[10433] = -this_block_jacobian[10434];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[10545] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[10546] = -this_block_jacobian[10545]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[10656] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[10655] = -this_block_jacobian[10656]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[10767] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[10768] = this_block_jacobian[10767]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[10879] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[10877] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[10878] = -this_block_jacobian[10877] - 
        this_block_jacobian[10879]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[10989] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[10988] = -this_block_jacobian[10989];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[11100] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[11101] = -this_block_jacobian[11100]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[11211] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[11210] = -this_block_jacobian[11211]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[11322] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[11323] = this_block_jacobian[11322]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[11434] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[11432] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[11433] = -this_block_jacobian[11432] - 
        this_block_jacobian[11434]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[11544] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[11543] = -this_block_jacobian[11544];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[11655] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[11656] = -this_block_jacobian[11655]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[11766] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[11765] = -this_block_jacobian[11766]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[11877] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[11878] = this_block_jacobian[11877]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[11989] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[11987] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[11988] = -this_block_jacobian[11987] - 
        this_block_jacobian[11989]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[12099] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[12098] = -this_block_jacobian[12099];//He+ : 9-alpha_(He++)ne
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
