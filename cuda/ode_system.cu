
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
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[115] = constants[2]*ne*this_block_state[116]
        -(constants[0]*ne + constants[1])*this_block_state[115]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[116] = -this_block_derivatives[115];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[117] = (constants[7]+constants[8])*ne*this_block_state[118] 
        - (constants[3]*ne+constants[4])*this_block_state[117];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[118] = constants[9]*ne*this_block_state[119] 
        + (constants[3]*ne+constants[4])*this_block_state[117]  
        - (constants[7]+constants[8])*ne*this_block_state[118] 
        - (constants[5]*ne+constants[6])*this_block_state[118];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[119] = (constants[5]*ne+constants[6])*this_block_state[118]
        -constants[9]*ne*this_block_state[119]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[120] = constants[2]*ne*this_block_state[121]
        -(constants[0]*ne + constants[1])*this_block_state[120]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[121] = -this_block_derivatives[120];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[122] = (constants[7]+constants[8])*ne*this_block_state[123] 
        - (constants[3]*ne+constants[4])*this_block_state[122];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[123] = constants[9]*ne*this_block_state[124] 
        + (constants[3]*ne+constants[4])*this_block_state[122]  
        - (constants[7]+constants[8])*ne*this_block_state[123] 
        - (constants[5]*ne+constants[6])*this_block_state[123];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[124] = (constants[5]*ne+constants[6])*this_block_state[123]
        -constants[9]*ne*this_block_state[124]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[125] = constants[2]*ne*this_block_state[126]
        -(constants[0]*ne + constants[1])*this_block_state[125]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[126] = -this_block_derivatives[125];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[127] = (constants[7]+constants[8])*ne*this_block_state[128] 
        - (constants[3]*ne+constants[4])*this_block_state[127];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[128] = constants[9]*ne*this_block_state[129] 
        + (constants[3]*ne+constants[4])*this_block_state[127]  
        - (constants[7]+constants[8])*ne*this_block_state[128] 
        - (constants[5]*ne+constants[6])*this_block_state[128];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[129] = (constants[5]*ne+constants[6])*this_block_state[128]
        -constants[9]*ne*this_block_state[129]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[130] = constants[2]*ne*this_block_state[131]
        -(constants[0]*ne + constants[1])*this_block_state[130]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[131] = -this_block_derivatives[130];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[132] = (constants[7]+constants[8])*ne*this_block_state[133] 
        - (constants[3]*ne+constants[4])*this_block_state[132];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[133] = constants[9]*ne*this_block_state[134] 
        + (constants[3]*ne+constants[4])*this_block_state[132]  
        - (constants[7]+constants[8])*ne*this_block_state[133] 
        - (constants[5]*ne+constants[6])*this_block_state[133];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[134] = (constants[5]*ne+constants[6])*this_block_state[133]
        -constants[9]*ne*this_block_state[134]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[135] = constants[2]*ne*this_block_state[136]
        -(constants[0]*ne + constants[1])*this_block_state[135]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[136] = -this_block_derivatives[135];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[137] = (constants[7]+constants[8])*ne*this_block_state[138] 
        - (constants[3]*ne+constants[4])*this_block_state[137];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[138] = constants[9]*ne*this_block_state[139] 
        + (constants[3]*ne+constants[4])*this_block_state[137]  
        - (constants[7]+constants[8])*ne*this_block_state[138] 
        - (constants[5]*ne+constants[6])*this_block_state[138];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[139] = (constants[5]*ne+constants[6])*this_block_state[138]
        -constants[9]*ne*this_block_state[139]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[140] = constants[2]*ne*this_block_state[141]
        -(constants[0]*ne + constants[1])*this_block_state[140]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[141] = -this_block_derivatives[140];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[142] = (constants[7]+constants[8])*ne*this_block_state[143] 
        - (constants[3]*ne+constants[4])*this_block_state[142];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[143] = constants[9]*ne*this_block_state[144] 
        + (constants[3]*ne+constants[4])*this_block_state[142]  
        - (constants[7]+constants[8])*ne*this_block_state[143] 
        - (constants[5]*ne+constants[6])*this_block_state[143];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[144] = (constants[5]*ne+constants[6])*this_block_state[143]
        -constants[9]*ne*this_block_state[144]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[145] = constants[2]*ne*this_block_state[146]
        -(constants[0]*ne + constants[1])*this_block_state[145]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[146] = -this_block_derivatives[145];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[147] = (constants[7]+constants[8])*ne*this_block_state[148] 
        - (constants[3]*ne+constants[4])*this_block_state[147];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[148] = constants[9]*ne*this_block_state[149] 
        + (constants[3]*ne+constants[4])*this_block_state[147]  
        - (constants[7]+constants[8])*ne*this_block_state[148] 
        - (constants[5]*ne+constants[6])*this_block_state[148];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[149] = (constants[5]*ne+constants[6])*this_block_state[148]
        -constants[9]*ne*this_block_state[149]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[150] = constants[2]*ne*this_block_state[151]
        -(constants[0]*ne + constants[1])*this_block_state[150]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[151] = -this_block_derivatives[150];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[152] = (constants[7]+constants[8])*ne*this_block_state[153] 
        - (constants[3]*ne+constants[4])*this_block_state[152];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[153] = constants[9]*ne*this_block_state[154] 
        + (constants[3]*ne+constants[4])*this_block_state[152]  
        - (constants[7]+constants[8])*ne*this_block_state[153] 
        - (constants[5]*ne+constants[6])*this_block_state[153];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[154] = (constants[5]*ne+constants[6])*this_block_state[153]
        -constants[9]*ne*this_block_state[154]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[155] = constants[2]*ne*this_block_state[156]
        -(constants[0]*ne + constants[1])*this_block_state[155]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[156] = -this_block_derivatives[155];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[157] = (constants[7]+constants[8])*ne*this_block_state[158] 
        - (constants[3]*ne+constants[4])*this_block_state[157];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[158] = constants[9]*ne*this_block_state[159] 
        + (constants[3]*ne+constants[4])*this_block_state[157]  
        - (constants[7]+constants[8])*ne*this_block_state[158] 
        - (constants[5]*ne+constants[6])*this_block_state[158];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[159] = (constants[5]*ne+constants[6])*this_block_state[158]
        -constants[9]*ne*this_block_state[159]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[160] = constants[2]*ne*this_block_state[161]
        -(constants[0]*ne + constants[1])*this_block_state[160]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[161] = -this_block_derivatives[160];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[162] = (constants[7]+constants[8])*ne*this_block_state[163] 
        - (constants[3]*ne+constants[4])*this_block_state[162];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[163] = constants[9]*ne*this_block_state[164] 
        + (constants[3]*ne+constants[4])*this_block_state[162]  
        - (constants[7]+constants[8])*ne*this_block_state[163] 
        - (constants[5]*ne+constants[6])*this_block_state[163];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[164] = (constants[5]*ne+constants[6])*this_block_state[163]
        -constants[9]*ne*this_block_state[164]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[165] = constants[2]*ne*this_block_state[166]
        -(constants[0]*ne + constants[1])*this_block_state[165]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[166] = -this_block_derivatives[165];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[167] = (constants[7]+constants[8])*ne*this_block_state[168] 
        - (constants[3]*ne+constants[4])*this_block_state[167];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[168] = constants[9]*ne*this_block_state[169] 
        + (constants[3]*ne+constants[4])*this_block_state[167]  
        - (constants[7]+constants[8])*ne*this_block_state[168] 
        - (constants[5]*ne+constants[6])*this_block_state[168];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[169] = (constants[5]*ne+constants[6])*this_block_state[168]
        -constants[9]*ne*this_block_state[169]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[170] = constants[2]*ne*this_block_state[171]
        -(constants[0]*ne + constants[1])*this_block_state[170]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[171] = -this_block_derivatives[170];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[172] = (constants[7]+constants[8])*ne*this_block_state[173] 
        - (constants[3]*ne+constants[4])*this_block_state[172];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[173] = constants[9]*ne*this_block_state[174] 
        + (constants[3]*ne+constants[4])*this_block_state[172]  
        - (constants[7]+constants[8])*ne*this_block_state[173] 
        - (constants[5]*ne+constants[6])*this_block_state[173];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[174] = (constants[5]*ne+constants[6])*this_block_state[173]
        -constants[9]*ne*this_block_state[174]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[175] = constants[2]*ne*this_block_state[176]
        -(constants[0]*ne + constants[1])*this_block_state[175]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[176] = -this_block_derivatives[175];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[177] = (constants[7]+constants[8])*ne*this_block_state[178] 
        - (constants[3]*ne+constants[4])*this_block_state[177];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[178] = constants[9]*ne*this_block_state[179] 
        + (constants[3]*ne+constants[4])*this_block_state[177]  
        - (constants[7]+constants[8])*ne*this_block_state[178] 
        - (constants[5]*ne+constants[6])*this_block_state[178];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[179] = (constants[5]*ne+constants[6])*this_block_state[178]
        -constants[9]*ne*this_block_state[179]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[180] = constants[2]*ne*this_block_state[181]
        -(constants[0]*ne + constants[1])*this_block_state[180]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[181] = -this_block_derivatives[180];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[182] = (constants[7]+constants[8])*ne*this_block_state[183] 
        - (constants[3]*ne+constants[4])*this_block_state[182];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[183] = constants[9]*ne*this_block_state[184] 
        + (constants[3]*ne+constants[4])*this_block_state[182]  
        - (constants[7]+constants[8])*ne*this_block_state[183] 
        - (constants[5]*ne+constants[6])*this_block_state[183];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[184] = (constants[5]*ne+constants[6])*this_block_state[183]
        -constants[9]*ne*this_block_state[184]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[185] = constants[2]*ne*this_block_state[186]
        -(constants[0]*ne + constants[1])*this_block_state[185]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[186] = -this_block_derivatives[185];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[187] = (constants[7]+constants[8])*ne*this_block_state[188] 
        - (constants[3]*ne+constants[4])*this_block_state[187];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[188] = constants[9]*ne*this_block_state[189] 
        + (constants[3]*ne+constants[4])*this_block_state[187]  
        - (constants[7]+constants[8])*ne*this_block_state[188] 
        - (constants[5]*ne+constants[6])*this_block_state[188];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[189] = (constants[5]*ne+constants[6])*this_block_state[188]
        -constants[9]*ne*this_block_state[189]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[190] = constants[2]*ne*this_block_state[191]
        -(constants[0]*ne + constants[1])*this_block_state[190]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[191] = -this_block_derivatives[190];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[192] = (constants[7]+constants[8])*ne*this_block_state[193] 
        - (constants[3]*ne+constants[4])*this_block_state[192];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[193] = constants[9]*ne*this_block_state[194] 
        + (constants[3]*ne+constants[4])*this_block_state[192]  
        - (constants[7]+constants[8])*ne*this_block_state[193] 
        - (constants[5]*ne+constants[6])*this_block_state[193];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[194] = (constants[5]*ne+constants[6])*this_block_state[193]
        -constants[9]*ne*this_block_state[194]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[195] = constants[2]*ne*this_block_state[196]
        -(constants[0]*ne + constants[1])*this_block_state[195]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[196] = -this_block_derivatives[195];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[197] = (constants[7]+constants[8])*ne*this_block_state[198] 
        - (constants[3]*ne+constants[4])*this_block_state[197];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[198] = constants[9]*ne*this_block_state[199] 
        + (constants[3]*ne+constants[4])*this_block_state[197]  
        - (constants[7]+constants[8])*ne*this_block_state[198] 
        - (constants[5]*ne+constants[6])*this_block_state[198];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[199] = (constants[5]*ne+constants[6])*this_block_state[198]
        -constants[9]*ne*this_block_state[199]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[200] = constants[2]*ne*this_block_state[201]
        -(constants[0]*ne + constants[1])*this_block_state[200]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[201] = -this_block_derivatives[200];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[202] = (constants[7]+constants[8])*ne*this_block_state[203] 
        - (constants[3]*ne+constants[4])*this_block_state[202];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[203] = constants[9]*ne*this_block_state[204] 
        + (constants[3]*ne+constants[4])*this_block_state[202]  
        - (constants[7]+constants[8])*ne*this_block_state[203] 
        - (constants[5]*ne+constants[6])*this_block_state[203];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[204] = (constants[5]*ne+constants[6])*this_block_state[203]
        -constants[9]*ne*this_block_state[204]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[205] = constants[2]*ne*this_block_state[206]
        -(constants[0]*ne + constants[1])*this_block_state[205]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[206] = -this_block_derivatives[205];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[207] = (constants[7]+constants[8])*ne*this_block_state[208] 
        - (constants[3]*ne+constants[4])*this_block_state[207];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[208] = constants[9]*ne*this_block_state[209] 
        + (constants[3]*ne+constants[4])*this_block_state[207]  
        - (constants[7]+constants[8])*ne*this_block_state[208] 
        - (constants[5]*ne+constants[6])*this_block_state[208];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[209] = (constants[5]*ne+constants[6])*this_block_state[208]
        -constants[9]*ne*this_block_state[209]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[210] = constants[2]*ne*this_block_state[211]
        -(constants[0]*ne + constants[1])*this_block_state[210]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[211] = -this_block_derivatives[210];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[212] = (constants[7]+constants[8])*ne*this_block_state[213] 
        - (constants[3]*ne+constants[4])*this_block_state[212];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[213] = constants[9]*ne*this_block_state[214] 
        + (constants[3]*ne+constants[4])*this_block_state[212]  
        - (constants[7]+constants[8])*ne*this_block_state[213] 
        - (constants[5]*ne+constants[6])*this_block_state[213];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[214] = (constants[5]*ne+constants[6])*this_block_state[213]
        -constants[9]*ne*this_block_state[214]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[215] = constants[2]*ne*this_block_state[216]
        -(constants[0]*ne + constants[1])*this_block_state[215]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[216] = -this_block_derivatives[215];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[217] = (constants[7]+constants[8])*ne*this_block_state[218] 
        - (constants[3]*ne+constants[4])*this_block_state[217];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[218] = constants[9]*ne*this_block_state[219] 
        + (constants[3]*ne+constants[4])*this_block_state[217]  
        - (constants[7]+constants[8])*ne*this_block_state[218] 
        - (constants[5]*ne+constants[6])*this_block_state[218];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[219] = (constants[5]*ne+constants[6])*this_block_state[218]
        -constants[9]*ne*this_block_state[219]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[220] = constants[2]*ne*this_block_state[221]
        -(constants[0]*ne + constants[1])*this_block_state[220]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[221] = -this_block_derivatives[220];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[222] = (constants[7]+constants[8])*ne*this_block_state[223] 
        - (constants[3]*ne+constants[4])*this_block_state[222];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[223] = constants[9]*ne*this_block_state[224] 
        + (constants[3]*ne+constants[4])*this_block_state[222]  
        - (constants[7]+constants[8])*ne*this_block_state[223] 
        - (constants[5]*ne+constants[6])*this_block_state[223];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[224] = (constants[5]*ne+constants[6])*this_block_state[223]
        -constants[9]*ne*this_block_state[224]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[225] = constants[2]*ne*this_block_state[226]
        -(constants[0]*ne + constants[1])*this_block_state[225]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[226] = -this_block_derivatives[225];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[227] = (constants[7]+constants[8])*ne*this_block_state[228] 
        - (constants[3]*ne+constants[4])*this_block_state[227];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[228] = constants[9]*ne*this_block_state[229] 
        + (constants[3]*ne+constants[4])*this_block_state[227]  
        - (constants[7]+constants[8])*ne*this_block_state[228] 
        - (constants[5]*ne+constants[6])*this_block_state[228];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[229] = (constants[5]*ne+constants[6])*this_block_state[228]
        -constants[9]*ne*this_block_state[229]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[230] = constants[2]*ne*this_block_state[231]
        -(constants[0]*ne + constants[1])*this_block_state[230]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[231] = -this_block_derivatives[230];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[232] = (constants[7]+constants[8])*ne*this_block_state[233] 
        - (constants[3]*ne+constants[4])*this_block_state[232];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[233] = constants[9]*ne*this_block_state[234] 
        + (constants[3]*ne+constants[4])*this_block_state[232]  
        - (constants[7]+constants[8])*ne*this_block_state[233] 
        - (constants[5]*ne+constants[6])*this_block_state[233];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[234] = (constants[5]*ne+constants[6])*this_block_state[233]
        -constants[9]*ne*this_block_state[234]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[235] = constants[2]*ne*this_block_state[236]
        -(constants[0]*ne + constants[1])*this_block_state[235]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[236] = -this_block_derivatives[235];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[237] = (constants[7]+constants[8])*ne*this_block_state[238] 
        - (constants[3]*ne+constants[4])*this_block_state[237];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[238] = constants[9]*ne*this_block_state[239] 
        + (constants[3]*ne+constants[4])*this_block_state[237]  
        - (constants[7]+constants[8])*ne*this_block_state[238] 
        - (constants[5]*ne+constants[6])*this_block_state[238];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[239] = (constants[5]*ne+constants[6])*this_block_state[238]
        -constants[9]*ne*this_block_state[239]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[240] = constants[2]*ne*this_block_state[241]
        -(constants[0]*ne + constants[1])*this_block_state[240]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[241] = -this_block_derivatives[240];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[242] = (constants[7]+constants[8])*ne*this_block_state[243] 
        - (constants[3]*ne+constants[4])*this_block_state[242];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[243] = constants[9]*ne*this_block_state[244] 
        + (constants[3]*ne+constants[4])*this_block_state[242]  
        - (constants[7]+constants[8])*ne*this_block_state[243] 
        - (constants[5]*ne+constants[6])*this_block_state[243];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[244] = (constants[5]*ne+constants[6])*this_block_state[243]
        -constants[9]*ne*this_block_state[244]; 
            // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[245] = constants[2]*ne*this_block_state[246]
        -(constants[0]*ne + constants[1])*this_block_state[245]; 
        
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[246] = -this_block_derivatives[245];
        
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[247] = (constants[7]+constants[8])*ne*this_block_state[248] 
        - (constants[3]*ne+constants[4])*this_block_state[247];
        
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[248] = constants[9]*ne*this_block_state[249] 
        + (constants[3]*ne+constants[4])*this_block_state[247]  
        - (constants[7]+constants[8])*ne*this_block_state[248] 
        - (constants[5]*ne+constants[6])*this_block_state[248];
        
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[249] = (constants[5]*ne+constants[6])*this_block_state[248]
        -constants[9]*ne*this_block_state[249]; 
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
    this_block_jacobian[251] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[250] = -this_block_jacobian[251]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[502] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[503] = this_block_jacobian[502]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[754] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[752] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[753] = -this_block_jacobian[752] - 
        this_block_jacobian[754]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[1004] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[1003] = -this_block_jacobian[1004];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[1255] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[1256] = -this_block_jacobian[1255]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[1506] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[1505] = -this_block_jacobian[1506]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[1757] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[1758] = this_block_jacobian[1757]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[2009] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[2007] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[2008] = -this_block_jacobian[2007] - 
        this_block_jacobian[2009]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[2259] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[2258] = -this_block_jacobian[2259];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[2510] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[2511] = -this_block_jacobian[2510]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[2761] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[2760] = -this_block_jacobian[2761]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[3012] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[3013] = this_block_jacobian[3012]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[3264] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[3262] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[3263] = -this_block_jacobian[3262] - 
        this_block_jacobian[3264]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[3514] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[3513] = -this_block_jacobian[3514];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[3765] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[3766] = -this_block_jacobian[3765]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[4016] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[4015] = -this_block_jacobian[4016]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[4267] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[4268] = this_block_jacobian[4267]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[4519] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[4517] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[4518] = -this_block_jacobian[4517] - 
        this_block_jacobian[4519]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[4769] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[4768] = -this_block_jacobian[4769];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[5020] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[5021] = -this_block_jacobian[5020]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[5271] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[5270] = -this_block_jacobian[5271]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[5522] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[5523] = this_block_jacobian[5522]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[5774] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[5772] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[5773] = -this_block_jacobian[5772] - 
        this_block_jacobian[5774]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[6024] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[6023] = -this_block_jacobian[6024];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[6275] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[6276] = -this_block_jacobian[6275]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[6526] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[6525] = -this_block_jacobian[6526]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[6777] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[6778] = this_block_jacobian[6777]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[7029] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[7027] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[7028] = -this_block_jacobian[7027] - 
        this_block_jacobian[7029]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[7279] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[7278] = -this_block_jacobian[7279];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[7530] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[7531] = -this_block_jacobian[7530]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[7781] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[7780] = -this_block_jacobian[7781]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[8032] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[8033] = this_block_jacobian[8032]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[8284] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[8282] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[8283] = -this_block_jacobian[8282] - 
        this_block_jacobian[8284]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[8534] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[8533] = -this_block_jacobian[8534];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[8785] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[8786] = -this_block_jacobian[8785]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[9036] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[9035] = -this_block_jacobian[9036]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[9287] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[9288] = this_block_jacobian[9287]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[9539] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[9537] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[9538] = -this_block_jacobian[9537] - 
        this_block_jacobian[9539]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[9789] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[9788] = -this_block_jacobian[9789];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[10040] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[10041] = -this_block_jacobian[10040]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[10291] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[10290] = -this_block_jacobian[10291]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[10542] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[10543] = this_block_jacobian[10542]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[10794] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[10792] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[10793] = -this_block_jacobian[10792] - 
        this_block_jacobian[10794]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[11044] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[11043] = -this_block_jacobian[11044];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[11295] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[11296] = -this_block_jacobian[11295]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[11546] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[11545] = -this_block_jacobian[11546]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[11797] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[11798] = this_block_jacobian[11797]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[12049] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[12047] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[12048] = -this_block_jacobian[12047] - 
        this_block_jacobian[12049]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[12299] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[12298] = -this_block_jacobian[12299];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[12550] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[12551] = -this_block_jacobian[12550]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[12801] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[12800] = -this_block_jacobian[12801]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[13052] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[13053] = this_block_jacobian[13052]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[13304] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[13302] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[13303] = -this_block_jacobian[13302] - 
        this_block_jacobian[13304]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[13554] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[13553] = -this_block_jacobian[13554];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[13805] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[13806] = -this_block_jacobian[13805]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[14056] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[14055] = -this_block_jacobian[14056]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[14307] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[14308] = this_block_jacobian[14307]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[14559] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[14557] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[14558] = -this_block_jacobian[14557] - 
        this_block_jacobian[14559]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[14809] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[14808] = -this_block_jacobian[14809];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[15060] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[15061] = -this_block_jacobian[15060]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[15311] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[15310] = -this_block_jacobian[15311]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[15562] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[15563] = this_block_jacobian[15562]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[15814] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[15812] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[15813] = -this_block_jacobian[15812] - 
        this_block_jacobian[15814]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[16064] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[16063] = -this_block_jacobian[16064];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[16315] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[16316] = -this_block_jacobian[16315]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[16566] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[16565] = -this_block_jacobian[16566]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[16817] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[16818] = this_block_jacobian[16817]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[17069] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[17067] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[17068] = -this_block_jacobian[17067] - 
        this_block_jacobian[17069]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[17319] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[17318] = -this_block_jacobian[17319];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[17570] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[17571] = -this_block_jacobian[17570]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[17821] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[17820] = -this_block_jacobian[17821]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[18072] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[18073] = this_block_jacobian[18072]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[18324] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[18322] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[18323] = -this_block_jacobian[18322] - 
        this_block_jacobian[18324]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[18574] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[18573] = -this_block_jacobian[18574];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[18825] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[18826] = -this_block_jacobian[18825]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[19076] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[19075] = -this_block_jacobian[19076]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[19327] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[19328] = this_block_jacobian[19327]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[19579] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[19577] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[19578] = -this_block_jacobian[19577] - 
        this_block_jacobian[19579]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[19829] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[19828] = -this_block_jacobian[19829];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[20080] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[20081] = -this_block_jacobian[20080]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[20331] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[20330] = -this_block_jacobian[20331]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[20582] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[20583] = this_block_jacobian[20582]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[20834] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[20832] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[20833] = -this_block_jacobian[20832] - 
        this_block_jacobian[20834]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[21084] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[21083] = -this_block_jacobian[21084];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[21335] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[21336] = -this_block_jacobian[21335]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[21586] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[21585] = -this_block_jacobian[21586]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[21837] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[21838] = this_block_jacobian[21837]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[22089] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[22087] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[22088] = -this_block_jacobian[22087] - 
        this_block_jacobian[22089]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[22339] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[22338] = -this_block_jacobian[22339];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[22590] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[22591] = -this_block_jacobian[22590]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[22841] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[22840] = -this_block_jacobian[22841]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[23092] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[23093] = this_block_jacobian[23092]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[23344] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[23342] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[23343] = -this_block_jacobian[23342] - 
        this_block_jacobian[23344]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[23594] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[23593] = -this_block_jacobian[23594];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[23845] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[23846] = -this_block_jacobian[23845]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[24096] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[24095] = -this_block_jacobian[24096]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[24347] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[24348] = this_block_jacobian[24347]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[24599] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[24597] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[24598] = -this_block_jacobian[24597] - 
        this_block_jacobian[24599]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[24849] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[24848] = -this_block_jacobian[24849];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[25100] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[25101] = -this_block_jacobian[25100]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[25351] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[25350] = -this_block_jacobian[25351]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[25602] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[25603] = this_block_jacobian[25602]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[25854] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[25852] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[25853] = -this_block_jacobian[25852] - 
        this_block_jacobian[25854]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[26104] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[26103] = -this_block_jacobian[26104];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[26355] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[26356] = -this_block_jacobian[26355]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[26606] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[26605] = -this_block_jacobian[26606]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[26857] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[26858] = this_block_jacobian[26857]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[27109] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[27107] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[27108] = -this_block_jacobian[27107] - 
        this_block_jacobian[27109]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[27359] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[27358] = -this_block_jacobian[27359];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[27610] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[27611] = -this_block_jacobian[27610]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[27861] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[27860] = -this_block_jacobian[27861]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[28112] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[28113] = this_block_jacobian[28112]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[28364] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[28362] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[28363] = -this_block_jacobian[28362] - 
        this_block_jacobian[28364]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[28614] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[28613] = -this_block_jacobian[28614];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[28865] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[28866] = -this_block_jacobian[28865]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[29116] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[29115] = -this_block_jacobian[29116]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[29367] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[29368] = this_block_jacobian[29367]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[29619] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[29617] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[29618] = -this_block_jacobian[29617] - 
        this_block_jacobian[29619]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[29869] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[29868] = -this_block_jacobian[29869];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[30120] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[30121] = -this_block_jacobian[30120]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[30371] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[30370] = -this_block_jacobian[30371]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[30622] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[30623] = this_block_jacobian[30622]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[30874] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[30872] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[30873] = -this_block_jacobian[30872] - 
        this_block_jacobian[30874]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[31124] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[31123] = -this_block_jacobian[31124];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[31375] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[31376] = -this_block_jacobian[31375]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[31626] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[31625] = -this_block_jacobian[31626]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[31877] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[31878] = this_block_jacobian[31877]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[32129] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[32127] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[32128] = -this_block_jacobian[32127] - 
        this_block_jacobian[32129]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[32379] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[32378] = -this_block_jacobian[32379];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[32630] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[32631] = -this_block_jacobian[32630]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[32881] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[32880] = -this_block_jacobian[32881]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[33132] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[33133] = this_block_jacobian[33132]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[33384] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[33382] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[33383] = -this_block_jacobian[33382] - 
        this_block_jacobian[33384]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[33634] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[33633] = -this_block_jacobian[33634];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[33885] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[33886] = -this_block_jacobian[33885]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[34136] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[34135] = -this_block_jacobian[34136]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[34387] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[34388] = this_block_jacobian[34387]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[34639] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[34637] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[34638] = -this_block_jacobian[34637] - 
        this_block_jacobian[34639]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[34889] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[34888] = -this_block_jacobian[34889];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[35140] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[35141] = -this_block_jacobian[35140]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[35391] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[35390] = -this_block_jacobian[35391]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[35642] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[35643] = this_block_jacobian[35642]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[35894] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[35892] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[35893] = -this_block_jacobian[35892] - 
        this_block_jacobian[35894]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[36144] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[36143] = -this_block_jacobian[36144];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[36395] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[36396] = -this_block_jacobian[36395]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[36646] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[36645] = -this_block_jacobian[36646]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[36897] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[36898] = this_block_jacobian[36897]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[37149] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[37147] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[37148] = -this_block_jacobian[37147] - 
        this_block_jacobian[37149]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[37399] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[37398] = -this_block_jacobian[37399];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[37650] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[37651] = -this_block_jacobian[37650]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[37901] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[37900] = -this_block_jacobian[37901]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[38152] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[38153] = this_block_jacobian[38152]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[38404] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[38402] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[38403] = -this_block_jacobian[38402] - 
        this_block_jacobian[38404]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[38654] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[38653] = -this_block_jacobian[38654];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[38905] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[38906] = -this_block_jacobian[38905]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[39156] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[39155] = -this_block_jacobian[39156]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[39407] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[39408] = this_block_jacobian[39407]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[39659] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[39657] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[39658] = -this_block_jacobian[39657] - 
        this_block_jacobian[39659]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[39909] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[39908] = -this_block_jacobian[39909];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[40160] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[40161] = -this_block_jacobian[40160]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[40411] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[40410] = -this_block_jacobian[40411]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[40662] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[40663] = this_block_jacobian[40662]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[40914] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[40912] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[40913] = -this_block_jacobian[40912] - 
        this_block_jacobian[40914]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[41164] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[41163] = -this_block_jacobian[41164];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[41415] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[41416] = -this_block_jacobian[41415]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[41666] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[41665] = -this_block_jacobian[41666]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[41917] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[41918] = this_block_jacobian[41917]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[42169] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[42167] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[42168] = -this_block_jacobian[42167] - 
        this_block_jacobian[42169]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[42419] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[42418] = -this_block_jacobian[42419];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[42670] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[42671] = -this_block_jacobian[42670]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[42921] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[42920] = -this_block_jacobian[42921]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[43172] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[43173] = this_block_jacobian[43172]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[43424] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[43422] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[43423] = -this_block_jacobian[43422] - 
        this_block_jacobian[43424]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[43674] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[43673] = -this_block_jacobian[43674];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[43925] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[43926] = -this_block_jacobian[43925]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[44176] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[44175] = -this_block_jacobian[44176]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[44427] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[44428] = this_block_jacobian[44427]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[44679] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[44677] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[44678] = -this_block_jacobian[44677] - 
        this_block_jacobian[44679]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[44929] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[44928] = -this_block_jacobian[44929];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[45180] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[45181] = -this_block_jacobian[45180]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[45431] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[45430] = -this_block_jacobian[45431]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[45682] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[45683] = this_block_jacobian[45682]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[45934] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[45932] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[45933] = -this_block_jacobian[45932] - 
        this_block_jacobian[45934]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[46184] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[46183] = -this_block_jacobian[46184];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[46435] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[46436] = -this_block_jacobian[46435]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[46686] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[46685] = -this_block_jacobian[46686]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[46937] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[46938] = this_block_jacobian[46937]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[47189] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[47187] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[47188] = -this_block_jacobian[47187] - 
        this_block_jacobian[47189]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[47439] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[47438] = -this_block_jacobian[47439];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[47690] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[47691] = -this_block_jacobian[47690]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[47941] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[47940] = -this_block_jacobian[47941]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[48192] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[48193] = this_block_jacobian[48192]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[48444] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[48442] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[48443] = -this_block_jacobian[48442] - 
        this_block_jacobian[48444]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[48694] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[48693] = -this_block_jacobian[48694];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[48945] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[48946] = -this_block_jacobian[48945]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[49196] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[49195] = -this_block_jacobian[49196]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[49447] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[49448] = this_block_jacobian[49447]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[49699] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[49697] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[49698] = -this_block_jacobian[49697] - 
        this_block_jacobian[49699]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[49949] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[49948] = -this_block_jacobian[49949];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[50200] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[50201] = -this_block_jacobian[50200]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[50451] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[50450] = -this_block_jacobian[50451]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[50702] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[50703] = this_block_jacobian[50702]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[50954] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[50952] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[50953] = -this_block_jacobian[50952] - 
        this_block_jacobian[50954]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[51204] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[51203] = -this_block_jacobian[51204];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[51455] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[51456] = -this_block_jacobian[51455]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[51706] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[51705] = -this_block_jacobian[51706]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[51957] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[51958] = this_block_jacobian[51957]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[52209] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[52207] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[52208] = -this_block_jacobian[52207] - 
        this_block_jacobian[52209]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[52459] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[52458] = -this_block_jacobian[52459];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[52710] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[52711] = -this_block_jacobian[52710]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[52961] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[52960] = -this_block_jacobian[52961]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[53212] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[53213] = this_block_jacobian[53212]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[53464] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[53462] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[53463] = -this_block_jacobian[53462] - 
        this_block_jacobian[53464]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[53714] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[53713] = -this_block_jacobian[53714];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[53965] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[53966] = -this_block_jacobian[53965]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[54216] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[54215] = -this_block_jacobian[54216]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[54467] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[54468] = this_block_jacobian[54467]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[54719] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[54717] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[54718] = -this_block_jacobian[54717] - 
        this_block_jacobian[54719]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[54969] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[54968] = -this_block_jacobian[54969];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[55220] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[55221] = -this_block_jacobian[55220]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[55471] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[55470] = -this_block_jacobian[55471]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[55722] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[55723] = this_block_jacobian[55722]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[55974] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[55972] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[55973] = -this_block_jacobian[55972] - 
        this_block_jacobian[55974]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[56224] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[56223] = -this_block_jacobian[56224];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[56475] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[56476] = -this_block_jacobian[56475]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[56726] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[56725] = -this_block_jacobian[56726]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[56977] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[56978] = this_block_jacobian[56977]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[57229] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[57227] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[57228] = -this_block_jacobian[57227] - 
        this_block_jacobian[57229]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[57479] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[57478] = -this_block_jacobian[57479];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[57730] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[57731] = -this_block_jacobian[57730]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[57981] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[57980] = -this_block_jacobian[57981]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[58232] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[58233] = this_block_jacobian[58232]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[58484] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[58482] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[58483] = -this_block_jacobian[58482] - 
        this_block_jacobian[58484]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[58734] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[58733] = -this_block_jacobian[58734];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[58985] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[58986] = -this_block_jacobian[58985]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[59236] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[59235] = -this_block_jacobian[59236]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[59487] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[59488] = this_block_jacobian[59487]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[59739] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[59737] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[59738] = -this_block_jacobian[59737] - 
        this_block_jacobian[59739]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[59989] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[59988] = -this_block_jacobian[59989];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[60240] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[60241] = -this_block_jacobian[60240]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[60491] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[60490] = -this_block_jacobian[60491]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[60742] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[60743] = this_block_jacobian[60742]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[60994] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[60992] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[60993] = -this_block_jacobian[60992] - 
        this_block_jacobian[60994]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[61244] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[61243] = -this_block_jacobian[61244];//He+ : 9-alpha_(He++)ne
           
    // H0
    this_block_jacobian[61495] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[61496] = -this_block_jacobian[61495]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    this_block_jacobian[61746] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[61745] = -this_block_jacobian[61746]; // H0 : 2-alpha_(H+)ne
        
    // He0
    this_block_jacobian[61997] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[61998] = this_block_jacobian[61997]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    this_block_jacobian[62249] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[62247] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[62248] = -this_block_jacobian[62247] - 
        this_block_jacobian[62249]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    this_block_jacobian[62499] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[62498] = -this_block_jacobian[62499];//He+ : 9-alpha_(He++)ne
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
