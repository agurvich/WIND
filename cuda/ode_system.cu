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
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[250] = constants[2]*ne*this_block_state[251]
        -(constants[0]*ne + constants[1])*this_block_state[250]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[251] = -this_block_derivatives[250];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[252] = (constants[7]+constants[8])*ne*this_block_state[253] 
        - (constants[3]*ne+constants[4])*this_block_state[252];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[253] = constants[9]*ne*this_block_state[254] 
        + (constants[3]*ne+constants[4])*this_block_state[252]  
        - (constants[7]+constants[8])*ne*this_block_state[253] 
        - (constants[5]*ne+constants[6])*this_block_state[253];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[254] = (constants[5]*ne+constants[6])*this_block_state[253]
        -constants[9]*ne*this_block_state[254]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[255] = constants[2]*ne*this_block_state[256]
        -(constants[0]*ne + constants[1])*this_block_state[255]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[256] = -this_block_derivatives[255];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[257] = (constants[7]+constants[8])*ne*this_block_state[258] 
        - (constants[3]*ne+constants[4])*this_block_state[257];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[258] = constants[9]*ne*this_block_state[259] 
        + (constants[3]*ne+constants[4])*this_block_state[257]  
        - (constants[7]+constants[8])*ne*this_block_state[258] 
        - (constants[5]*ne+constants[6])*this_block_state[258];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[259] = (constants[5]*ne+constants[6])*this_block_state[258]
        -constants[9]*ne*this_block_state[259]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[260] = constants[2]*ne*this_block_state[261]
        -(constants[0]*ne + constants[1])*this_block_state[260]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[261] = -this_block_derivatives[260];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[262] = (constants[7]+constants[8])*ne*this_block_state[263] 
        - (constants[3]*ne+constants[4])*this_block_state[262];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[263] = constants[9]*ne*this_block_state[264] 
        + (constants[3]*ne+constants[4])*this_block_state[262]  
        - (constants[7]+constants[8])*ne*this_block_state[263] 
        - (constants[5]*ne+constants[6])*this_block_state[263];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[264] = (constants[5]*ne+constants[6])*this_block_state[263]
        -constants[9]*ne*this_block_state[264]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[265] = constants[2]*ne*this_block_state[266]
        -(constants[0]*ne + constants[1])*this_block_state[265]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[266] = -this_block_derivatives[265];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[267] = (constants[7]+constants[8])*ne*this_block_state[268] 
        - (constants[3]*ne+constants[4])*this_block_state[267];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[268] = constants[9]*ne*this_block_state[269] 
        + (constants[3]*ne+constants[4])*this_block_state[267]  
        - (constants[7]+constants[8])*ne*this_block_state[268] 
        - (constants[5]*ne+constants[6])*this_block_state[268];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[269] = (constants[5]*ne+constants[6])*this_block_state[268]
        -constants[9]*ne*this_block_state[269]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[270] = constants[2]*ne*this_block_state[271]
        -(constants[0]*ne + constants[1])*this_block_state[270]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[271] = -this_block_derivatives[270];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[272] = (constants[7]+constants[8])*ne*this_block_state[273] 
        - (constants[3]*ne+constants[4])*this_block_state[272];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[273] = constants[9]*ne*this_block_state[274] 
        + (constants[3]*ne+constants[4])*this_block_state[272]  
        - (constants[7]+constants[8])*ne*this_block_state[273] 
        - (constants[5]*ne+constants[6])*this_block_state[273];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[274] = (constants[5]*ne+constants[6])*this_block_state[273]
        -constants[9]*ne*this_block_state[274]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[275] = constants[2]*ne*this_block_state[276]
        -(constants[0]*ne + constants[1])*this_block_state[275]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[276] = -this_block_derivatives[275];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[277] = (constants[7]+constants[8])*ne*this_block_state[278] 
        - (constants[3]*ne+constants[4])*this_block_state[277];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[278] = constants[9]*ne*this_block_state[279] 
        + (constants[3]*ne+constants[4])*this_block_state[277]  
        - (constants[7]+constants[8])*ne*this_block_state[278] 
        - (constants[5]*ne+constants[6])*this_block_state[278];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[279] = (constants[5]*ne+constants[6])*this_block_state[278]
        -constants[9]*ne*this_block_state[279]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[280] = constants[2]*ne*this_block_state[281]
        -(constants[0]*ne + constants[1])*this_block_state[280]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[281] = -this_block_derivatives[280];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[282] = (constants[7]+constants[8])*ne*this_block_state[283] 
        - (constants[3]*ne+constants[4])*this_block_state[282];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[283] = constants[9]*ne*this_block_state[284] 
        + (constants[3]*ne+constants[4])*this_block_state[282]  
        - (constants[7]+constants[8])*ne*this_block_state[283] 
        - (constants[5]*ne+constants[6])*this_block_state[283];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[284] = (constants[5]*ne+constants[6])*this_block_state[283]
        -constants[9]*ne*this_block_state[284]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[285] = constants[2]*ne*this_block_state[286]
        -(constants[0]*ne + constants[1])*this_block_state[285]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[286] = -this_block_derivatives[285];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[287] = (constants[7]+constants[8])*ne*this_block_state[288] 
        - (constants[3]*ne+constants[4])*this_block_state[287];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[288] = constants[9]*ne*this_block_state[289] 
        + (constants[3]*ne+constants[4])*this_block_state[287]  
        - (constants[7]+constants[8])*ne*this_block_state[288] 
        - (constants[5]*ne+constants[6])*this_block_state[288];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[289] = (constants[5]*ne+constants[6])*this_block_state[288]
        -constants[9]*ne*this_block_state[289]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[290] = constants[2]*ne*this_block_state[291]
        -(constants[0]*ne + constants[1])*this_block_state[290]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[291] = -this_block_derivatives[290];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[292] = (constants[7]+constants[8])*ne*this_block_state[293] 
        - (constants[3]*ne+constants[4])*this_block_state[292];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[293] = constants[9]*ne*this_block_state[294] 
        + (constants[3]*ne+constants[4])*this_block_state[292]  
        - (constants[7]+constants[8])*ne*this_block_state[293] 
        - (constants[5]*ne+constants[6])*this_block_state[293];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[294] = (constants[5]*ne+constants[6])*this_block_state[293]
        -constants[9]*ne*this_block_state[294]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[295] = constants[2]*ne*this_block_state[296]
        -(constants[0]*ne + constants[1])*this_block_state[295]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[296] = -this_block_derivatives[295];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[297] = (constants[7]+constants[8])*ne*this_block_state[298] 
        - (constants[3]*ne+constants[4])*this_block_state[297];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[298] = constants[9]*ne*this_block_state[299] 
        + (constants[3]*ne+constants[4])*this_block_state[297]  
        - (constants[7]+constants[8])*ne*this_block_state[298] 
        - (constants[5]*ne+constants[6])*this_block_state[298];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[299] = (constants[5]*ne+constants[6])*this_block_state[298]
        -constants[9]*ne*this_block_state[299]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[300] = constants[2]*ne*this_block_state[301]
        -(constants[0]*ne + constants[1])*this_block_state[300]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[301] = -this_block_derivatives[300];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[302] = (constants[7]+constants[8])*ne*this_block_state[303] 
        - (constants[3]*ne+constants[4])*this_block_state[302];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[303] = constants[9]*ne*this_block_state[304] 
        + (constants[3]*ne+constants[4])*this_block_state[302]  
        - (constants[7]+constants[8])*ne*this_block_state[303] 
        - (constants[5]*ne+constants[6])*this_block_state[303];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[304] = (constants[5]*ne+constants[6])*this_block_state[303]
        -constants[9]*ne*this_block_state[304]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[305] = constants[2]*ne*this_block_state[306]
        -(constants[0]*ne + constants[1])*this_block_state[305]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[306] = -this_block_derivatives[305];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[307] = (constants[7]+constants[8])*ne*this_block_state[308] 
        - (constants[3]*ne+constants[4])*this_block_state[307];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[308] = constants[9]*ne*this_block_state[309] 
        + (constants[3]*ne+constants[4])*this_block_state[307]  
        - (constants[7]+constants[8])*ne*this_block_state[308] 
        - (constants[5]*ne+constants[6])*this_block_state[308];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[309] = (constants[5]*ne+constants[6])*this_block_state[308]
        -constants[9]*ne*this_block_state[309]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[310] = constants[2]*ne*this_block_state[311]
        -(constants[0]*ne + constants[1])*this_block_state[310]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[311] = -this_block_derivatives[310];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[312] = (constants[7]+constants[8])*ne*this_block_state[313] 
        - (constants[3]*ne+constants[4])*this_block_state[312];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[313] = constants[9]*ne*this_block_state[314] 
        + (constants[3]*ne+constants[4])*this_block_state[312]  
        - (constants[7]+constants[8])*ne*this_block_state[313] 
        - (constants[5]*ne+constants[6])*this_block_state[313];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[314] = (constants[5]*ne+constants[6])*this_block_state[313]
        -constants[9]*ne*this_block_state[314]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[315] = constants[2]*ne*this_block_state[316]
        -(constants[0]*ne + constants[1])*this_block_state[315]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[316] = -this_block_derivatives[315];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[317] = (constants[7]+constants[8])*ne*this_block_state[318] 
        - (constants[3]*ne+constants[4])*this_block_state[317];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[318] = constants[9]*ne*this_block_state[319] 
        + (constants[3]*ne+constants[4])*this_block_state[317]  
        - (constants[7]+constants[8])*ne*this_block_state[318] 
        - (constants[5]*ne+constants[6])*this_block_state[318];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[319] = (constants[5]*ne+constants[6])*this_block_state[318]
        -constants[9]*ne*this_block_state[319]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[320] = constants[2]*ne*this_block_state[321]
        -(constants[0]*ne + constants[1])*this_block_state[320]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[321] = -this_block_derivatives[320];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[322] = (constants[7]+constants[8])*ne*this_block_state[323] 
        - (constants[3]*ne+constants[4])*this_block_state[322];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[323] = constants[9]*ne*this_block_state[324] 
        + (constants[3]*ne+constants[4])*this_block_state[322]  
        - (constants[7]+constants[8])*ne*this_block_state[323] 
        - (constants[5]*ne+constants[6])*this_block_state[323];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[324] = (constants[5]*ne+constants[6])*this_block_state[323]
        -constants[9]*ne*this_block_state[324]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[325] = constants[2]*ne*this_block_state[326]
        -(constants[0]*ne + constants[1])*this_block_state[325]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[326] = -this_block_derivatives[325];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[327] = (constants[7]+constants[8])*ne*this_block_state[328] 
        - (constants[3]*ne+constants[4])*this_block_state[327];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[328] = constants[9]*ne*this_block_state[329] 
        + (constants[3]*ne+constants[4])*this_block_state[327]  
        - (constants[7]+constants[8])*ne*this_block_state[328] 
        - (constants[5]*ne+constants[6])*this_block_state[328];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[329] = (constants[5]*ne+constants[6])*this_block_state[328]
        -constants[9]*ne*this_block_state[329]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[330] = constants[2]*ne*this_block_state[331]
        -(constants[0]*ne + constants[1])*this_block_state[330]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[331] = -this_block_derivatives[330];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[332] = (constants[7]+constants[8])*ne*this_block_state[333] 
        - (constants[3]*ne+constants[4])*this_block_state[332];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[333] = constants[9]*ne*this_block_state[334] 
        + (constants[3]*ne+constants[4])*this_block_state[332]  
        - (constants[7]+constants[8])*ne*this_block_state[333] 
        - (constants[5]*ne+constants[6])*this_block_state[333];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[334] = (constants[5]*ne+constants[6])*this_block_state[333]
        -constants[9]*ne*this_block_state[334]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[335] = constants[2]*ne*this_block_state[336]
        -(constants[0]*ne + constants[1])*this_block_state[335]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[336] = -this_block_derivatives[335];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[337] = (constants[7]+constants[8])*ne*this_block_state[338] 
        - (constants[3]*ne+constants[4])*this_block_state[337];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[338] = constants[9]*ne*this_block_state[339] 
        + (constants[3]*ne+constants[4])*this_block_state[337]  
        - (constants[7]+constants[8])*ne*this_block_state[338] 
        - (constants[5]*ne+constants[6])*this_block_state[338];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[339] = (constants[5]*ne+constants[6])*this_block_state[338]
        -constants[9]*ne*this_block_state[339]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[340] = constants[2]*ne*this_block_state[341]
        -(constants[0]*ne + constants[1])*this_block_state[340]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[341] = -this_block_derivatives[340];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[342] = (constants[7]+constants[8])*ne*this_block_state[343] 
        - (constants[3]*ne+constants[4])*this_block_state[342];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[343] = constants[9]*ne*this_block_state[344] 
        + (constants[3]*ne+constants[4])*this_block_state[342]  
        - (constants[7]+constants[8])*ne*this_block_state[343] 
        - (constants[5]*ne+constants[6])*this_block_state[343];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[344] = (constants[5]*ne+constants[6])*this_block_state[343]
        -constants[9]*ne*this_block_state[344]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[345] = constants[2]*ne*this_block_state[346]
        -(constants[0]*ne + constants[1])*this_block_state[345]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[346] = -this_block_derivatives[345];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[347] = (constants[7]+constants[8])*ne*this_block_state[348] 
        - (constants[3]*ne+constants[4])*this_block_state[347];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[348] = constants[9]*ne*this_block_state[349] 
        + (constants[3]*ne+constants[4])*this_block_state[347]  
        - (constants[7]+constants[8])*ne*this_block_state[348] 
        - (constants[5]*ne+constants[6])*this_block_state[348];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[349] = (constants[5]*ne+constants[6])*this_block_state[348]
        -constants[9]*ne*this_block_state[349]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[350] = constants[2]*ne*this_block_state[351]
        -(constants[0]*ne + constants[1])*this_block_state[350]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[351] = -this_block_derivatives[350];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[352] = (constants[7]+constants[8])*ne*this_block_state[353] 
        - (constants[3]*ne+constants[4])*this_block_state[352];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[353] = constants[9]*ne*this_block_state[354] 
        + (constants[3]*ne+constants[4])*this_block_state[352]  
        - (constants[7]+constants[8])*ne*this_block_state[353] 
        - (constants[5]*ne+constants[6])*this_block_state[353];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[354] = (constants[5]*ne+constants[6])*this_block_state[353]
        -constants[9]*ne*this_block_state[354]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[355] = constants[2]*ne*this_block_state[356]
        -(constants[0]*ne + constants[1])*this_block_state[355]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[356] = -this_block_derivatives[355];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[357] = (constants[7]+constants[8])*ne*this_block_state[358] 
        - (constants[3]*ne+constants[4])*this_block_state[357];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[358] = constants[9]*ne*this_block_state[359] 
        + (constants[3]*ne+constants[4])*this_block_state[357]  
        - (constants[7]+constants[8])*ne*this_block_state[358] 
        - (constants[5]*ne+constants[6])*this_block_state[358];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[359] = (constants[5]*ne+constants[6])*this_block_state[358]
        -constants[9]*ne*this_block_state[359]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[360] = constants[2]*ne*this_block_state[361]
        -(constants[0]*ne + constants[1])*this_block_state[360]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[361] = -this_block_derivatives[360];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[362] = (constants[7]+constants[8])*ne*this_block_state[363] 
        - (constants[3]*ne+constants[4])*this_block_state[362];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[363] = constants[9]*ne*this_block_state[364] 
        + (constants[3]*ne+constants[4])*this_block_state[362]  
        - (constants[7]+constants[8])*ne*this_block_state[363] 
        - (constants[5]*ne+constants[6])*this_block_state[363];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[364] = (constants[5]*ne+constants[6])*this_block_state[363]
        -constants[9]*ne*this_block_state[364]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[365] = constants[2]*ne*this_block_state[366]
        -(constants[0]*ne + constants[1])*this_block_state[365]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[366] = -this_block_derivatives[365];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[367] = (constants[7]+constants[8])*ne*this_block_state[368] 
        - (constants[3]*ne+constants[4])*this_block_state[367];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[368] = constants[9]*ne*this_block_state[369] 
        + (constants[3]*ne+constants[4])*this_block_state[367]  
        - (constants[7]+constants[8])*ne*this_block_state[368] 
        - (constants[5]*ne+constants[6])*this_block_state[368];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[369] = (constants[5]*ne+constants[6])*this_block_state[368]
        -constants[9]*ne*this_block_state[369]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[370] = constants[2]*ne*this_block_state[371]
        -(constants[0]*ne + constants[1])*this_block_state[370]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[371] = -this_block_derivatives[370];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[372] = (constants[7]+constants[8])*ne*this_block_state[373] 
        - (constants[3]*ne+constants[4])*this_block_state[372];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[373] = constants[9]*ne*this_block_state[374] 
        + (constants[3]*ne+constants[4])*this_block_state[372]  
        - (constants[7]+constants[8])*ne*this_block_state[373] 
        - (constants[5]*ne+constants[6])*this_block_state[373];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[374] = (constants[5]*ne+constants[6])*this_block_state[373]
        -constants[9]*ne*this_block_state[374]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[375] = constants[2]*ne*this_block_state[376]
        -(constants[0]*ne + constants[1])*this_block_state[375]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[376] = -this_block_derivatives[375];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[377] = (constants[7]+constants[8])*ne*this_block_state[378] 
        - (constants[3]*ne+constants[4])*this_block_state[377];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[378] = constants[9]*ne*this_block_state[379] 
        + (constants[3]*ne+constants[4])*this_block_state[377]  
        - (constants[7]+constants[8])*ne*this_block_state[378] 
        - (constants[5]*ne+constants[6])*this_block_state[378];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[379] = (constants[5]*ne+constants[6])*this_block_state[378]
        -constants[9]*ne*this_block_state[379]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[380] = constants[2]*ne*this_block_state[381]
        -(constants[0]*ne + constants[1])*this_block_state[380]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[381] = -this_block_derivatives[380];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[382] = (constants[7]+constants[8])*ne*this_block_state[383] 
        - (constants[3]*ne+constants[4])*this_block_state[382];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[383] = constants[9]*ne*this_block_state[384] 
        + (constants[3]*ne+constants[4])*this_block_state[382]  
        - (constants[7]+constants[8])*ne*this_block_state[383] 
        - (constants[5]*ne+constants[6])*this_block_state[383];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[384] = (constants[5]*ne+constants[6])*this_block_state[383]
        -constants[9]*ne*this_block_state[384]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[385] = constants[2]*ne*this_block_state[386]
        -(constants[0]*ne + constants[1])*this_block_state[385]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[386] = -this_block_derivatives[385];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[387] = (constants[7]+constants[8])*ne*this_block_state[388] 
        - (constants[3]*ne+constants[4])*this_block_state[387];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[388] = constants[9]*ne*this_block_state[389] 
        + (constants[3]*ne+constants[4])*this_block_state[387]  
        - (constants[7]+constants[8])*ne*this_block_state[388] 
        - (constants[5]*ne+constants[6])*this_block_state[388];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[389] = (constants[5]*ne+constants[6])*this_block_state[388]
        -constants[9]*ne*this_block_state[389]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[390] = constants[2]*ne*this_block_state[391]
        -(constants[0]*ne + constants[1])*this_block_state[390]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[391] = -this_block_derivatives[390];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[392] = (constants[7]+constants[8])*ne*this_block_state[393] 
        - (constants[3]*ne+constants[4])*this_block_state[392];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[393] = constants[9]*ne*this_block_state[394] 
        + (constants[3]*ne+constants[4])*this_block_state[392]  
        - (constants[7]+constants[8])*ne*this_block_state[393] 
        - (constants[5]*ne+constants[6])*this_block_state[393];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[394] = (constants[5]*ne+constants[6])*this_block_state[393]
        -constants[9]*ne*this_block_state[394]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[395] = constants[2]*ne*this_block_state[396]
        -(constants[0]*ne + constants[1])*this_block_state[395]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[396] = -this_block_derivatives[395];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[397] = (constants[7]+constants[8])*ne*this_block_state[398] 
        - (constants[3]*ne+constants[4])*this_block_state[397];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[398] = constants[9]*ne*this_block_state[399] 
        + (constants[3]*ne+constants[4])*this_block_state[397]  
        - (constants[7]+constants[8])*ne*this_block_state[398] 
        - (constants[5]*ne+constants[6])*this_block_state[398];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[399] = (constants[5]*ne+constants[6])*this_block_state[398]
        -constants[9]*ne*this_block_state[399]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[400] = constants[2]*ne*this_block_state[401]
        -(constants[0]*ne + constants[1])*this_block_state[400]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[401] = -this_block_derivatives[400];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[402] = (constants[7]+constants[8])*ne*this_block_state[403] 
        - (constants[3]*ne+constants[4])*this_block_state[402];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[403] = constants[9]*ne*this_block_state[404] 
        + (constants[3]*ne+constants[4])*this_block_state[402]  
        - (constants[7]+constants[8])*ne*this_block_state[403] 
        - (constants[5]*ne+constants[6])*this_block_state[403];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[404] = (constants[5]*ne+constants[6])*this_block_state[403]
        -constants[9]*ne*this_block_state[404]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[405] = constants[2]*ne*this_block_state[406]
        -(constants[0]*ne + constants[1])*this_block_state[405]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[406] = -this_block_derivatives[405];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[407] = (constants[7]+constants[8])*ne*this_block_state[408] 
        - (constants[3]*ne+constants[4])*this_block_state[407];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[408] = constants[9]*ne*this_block_state[409] 
        + (constants[3]*ne+constants[4])*this_block_state[407]  
        - (constants[7]+constants[8])*ne*this_block_state[408] 
        - (constants[5]*ne+constants[6])*this_block_state[408];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[409] = (constants[5]*ne+constants[6])*this_block_state[408]
        -constants[9]*ne*this_block_state[409]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[410] = constants[2]*ne*this_block_state[411]
        -(constants[0]*ne + constants[1])*this_block_state[410]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[411] = -this_block_derivatives[410];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[412] = (constants[7]+constants[8])*ne*this_block_state[413] 
        - (constants[3]*ne+constants[4])*this_block_state[412];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[413] = constants[9]*ne*this_block_state[414] 
        + (constants[3]*ne+constants[4])*this_block_state[412]  
        - (constants[7]+constants[8])*ne*this_block_state[413] 
        - (constants[5]*ne+constants[6])*this_block_state[413];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[414] = (constants[5]*ne+constants[6])*this_block_state[413]
        -constants[9]*ne*this_block_state[414]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[415] = constants[2]*ne*this_block_state[416]
        -(constants[0]*ne + constants[1])*this_block_state[415]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[416] = -this_block_derivatives[415];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[417] = (constants[7]+constants[8])*ne*this_block_state[418] 
        - (constants[3]*ne+constants[4])*this_block_state[417];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[418] = constants[9]*ne*this_block_state[419] 
        + (constants[3]*ne+constants[4])*this_block_state[417]  
        - (constants[7]+constants[8])*ne*this_block_state[418] 
        - (constants[5]*ne+constants[6])*this_block_state[418];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[419] = (constants[5]*ne+constants[6])*this_block_state[418]
        -constants[9]*ne*this_block_state[419]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[420] = constants[2]*ne*this_block_state[421]
        -(constants[0]*ne + constants[1])*this_block_state[420]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[421] = -this_block_derivatives[420];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[422] = (constants[7]+constants[8])*ne*this_block_state[423] 
        - (constants[3]*ne+constants[4])*this_block_state[422];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[423] = constants[9]*ne*this_block_state[424] 
        + (constants[3]*ne+constants[4])*this_block_state[422]  
        - (constants[7]+constants[8])*ne*this_block_state[423] 
        - (constants[5]*ne+constants[6])*this_block_state[423];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[424] = (constants[5]*ne+constants[6])*this_block_state[423]
        -constants[9]*ne*this_block_state[424]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[425] = constants[2]*ne*this_block_state[426]
        -(constants[0]*ne + constants[1])*this_block_state[425]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[426] = -this_block_derivatives[425];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[427] = (constants[7]+constants[8])*ne*this_block_state[428] 
        - (constants[3]*ne+constants[4])*this_block_state[427];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[428] = constants[9]*ne*this_block_state[429] 
        + (constants[3]*ne+constants[4])*this_block_state[427]  
        - (constants[7]+constants[8])*ne*this_block_state[428] 
        - (constants[5]*ne+constants[6])*this_block_state[428];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[429] = (constants[5]*ne+constants[6])*this_block_state[428]
        -constants[9]*ne*this_block_state[429]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[430] = constants[2]*ne*this_block_state[431]
        -(constants[0]*ne + constants[1])*this_block_state[430]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[431] = -this_block_derivatives[430];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[432] = (constants[7]+constants[8])*ne*this_block_state[433] 
        - (constants[3]*ne+constants[4])*this_block_state[432];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[433] = constants[9]*ne*this_block_state[434] 
        + (constants[3]*ne+constants[4])*this_block_state[432]  
        - (constants[7]+constants[8])*ne*this_block_state[433] 
        - (constants[5]*ne+constants[6])*this_block_state[433];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[434] = (constants[5]*ne+constants[6])*this_block_state[433]
        -constants[9]*ne*this_block_state[434]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[435] = constants[2]*ne*this_block_state[436]
        -(constants[0]*ne + constants[1])*this_block_state[435]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[436] = -this_block_derivatives[435];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[437] = (constants[7]+constants[8])*ne*this_block_state[438] 
        - (constants[3]*ne+constants[4])*this_block_state[437];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[438] = constants[9]*ne*this_block_state[439] 
        + (constants[3]*ne+constants[4])*this_block_state[437]  
        - (constants[7]+constants[8])*ne*this_block_state[438] 
        - (constants[5]*ne+constants[6])*this_block_state[438];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[439] = (constants[5]*ne+constants[6])*this_block_state[438]
        -constants[9]*ne*this_block_state[439]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[440] = constants[2]*ne*this_block_state[441]
        -(constants[0]*ne + constants[1])*this_block_state[440]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[441] = -this_block_derivatives[440];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[442] = (constants[7]+constants[8])*ne*this_block_state[443] 
        - (constants[3]*ne+constants[4])*this_block_state[442];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[443] = constants[9]*ne*this_block_state[444] 
        + (constants[3]*ne+constants[4])*this_block_state[442]  
        - (constants[7]+constants[8])*ne*this_block_state[443] 
        - (constants[5]*ne+constants[6])*this_block_state[443];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[444] = (constants[5]*ne+constants[6])*this_block_state[443]
        -constants[9]*ne*this_block_state[444]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[445] = constants[2]*ne*this_block_state[446]
        -(constants[0]*ne + constants[1])*this_block_state[445]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[446] = -this_block_derivatives[445];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[447] = (constants[7]+constants[8])*ne*this_block_state[448] 
        - (constants[3]*ne+constants[4])*this_block_state[447];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[448] = constants[9]*ne*this_block_state[449] 
        + (constants[3]*ne+constants[4])*this_block_state[447]  
        - (constants[7]+constants[8])*ne*this_block_state[448] 
        - (constants[5]*ne+constants[6])*this_block_state[448];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[449] = (constants[5]*ne+constants[6])*this_block_state[448]
        -constants[9]*ne*this_block_state[449]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[450] = constants[2]*ne*this_block_state[451]
        -(constants[0]*ne + constants[1])*this_block_state[450]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[451] = -this_block_derivatives[450];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[452] = (constants[7]+constants[8])*ne*this_block_state[453] 
        - (constants[3]*ne+constants[4])*this_block_state[452];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[453] = constants[9]*ne*this_block_state[454] 
        + (constants[3]*ne+constants[4])*this_block_state[452]  
        - (constants[7]+constants[8])*ne*this_block_state[453] 
        - (constants[5]*ne+constants[6])*this_block_state[453];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[454] = (constants[5]*ne+constants[6])*this_block_state[453]
        -constants[9]*ne*this_block_state[454]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[455] = constants[2]*ne*this_block_state[456]
        -(constants[0]*ne + constants[1])*this_block_state[455]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[456] = -this_block_derivatives[455];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[457] = (constants[7]+constants[8])*ne*this_block_state[458] 
        - (constants[3]*ne+constants[4])*this_block_state[457];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[458] = constants[9]*ne*this_block_state[459] 
        + (constants[3]*ne+constants[4])*this_block_state[457]  
        - (constants[7]+constants[8])*ne*this_block_state[458] 
        - (constants[5]*ne+constants[6])*this_block_state[458];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[459] = (constants[5]*ne+constants[6])*this_block_state[458]
        -constants[9]*ne*this_block_state[459]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[460] = constants[2]*ne*this_block_state[461]
        -(constants[0]*ne + constants[1])*this_block_state[460]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[461] = -this_block_derivatives[460];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[462] = (constants[7]+constants[8])*ne*this_block_state[463] 
        - (constants[3]*ne+constants[4])*this_block_state[462];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[463] = constants[9]*ne*this_block_state[464] 
        + (constants[3]*ne+constants[4])*this_block_state[462]  
        - (constants[7]+constants[8])*ne*this_block_state[463] 
        - (constants[5]*ne+constants[6])*this_block_state[463];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[464] = (constants[5]*ne+constants[6])*this_block_state[463]
        -constants[9]*ne*this_block_state[464]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[465] = constants[2]*ne*this_block_state[466]
        -(constants[0]*ne + constants[1])*this_block_state[465]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[466] = -this_block_derivatives[465];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[467] = (constants[7]+constants[8])*ne*this_block_state[468] 
        - (constants[3]*ne+constants[4])*this_block_state[467];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[468] = constants[9]*ne*this_block_state[469] 
        + (constants[3]*ne+constants[4])*this_block_state[467]  
        - (constants[7]+constants[8])*ne*this_block_state[468] 
        - (constants[5]*ne+constants[6])*this_block_state[468];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[469] = (constants[5]*ne+constants[6])*this_block_state[468]
        -constants[9]*ne*this_block_state[469]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[470] = constants[2]*ne*this_block_state[471]
        -(constants[0]*ne + constants[1])*this_block_state[470]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[471] = -this_block_derivatives[470];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[472] = (constants[7]+constants[8])*ne*this_block_state[473] 
        - (constants[3]*ne+constants[4])*this_block_state[472];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[473] = constants[9]*ne*this_block_state[474] 
        + (constants[3]*ne+constants[4])*this_block_state[472]  
        - (constants[7]+constants[8])*ne*this_block_state[473] 
        - (constants[5]*ne+constants[6])*this_block_state[473];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[474] = (constants[5]*ne+constants[6])*this_block_state[473]
        -constants[9]*ne*this_block_state[474]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[475] = constants[2]*ne*this_block_state[476]
        -(constants[0]*ne + constants[1])*this_block_state[475]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[476] = -this_block_derivatives[475];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[477] = (constants[7]+constants[8])*ne*this_block_state[478] 
        - (constants[3]*ne+constants[4])*this_block_state[477];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[478] = constants[9]*ne*this_block_state[479] 
        + (constants[3]*ne+constants[4])*this_block_state[477]  
        - (constants[7]+constants[8])*ne*this_block_state[478] 
        - (constants[5]*ne+constants[6])*this_block_state[478];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[479] = (constants[5]*ne+constants[6])*this_block_state[478]
        -constants[9]*ne*this_block_state[479]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[480] = constants[2]*ne*this_block_state[481]
        -(constants[0]*ne + constants[1])*this_block_state[480]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[481] = -this_block_derivatives[480];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[482] = (constants[7]+constants[8])*ne*this_block_state[483] 
        - (constants[3]*ne+constants[4])*this_block_state[482];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[483] = constants[9]*ne*this_block_state[484] 
        + (constants[3]*ne+constants[4])*this_block_state[482]  
        - (constants[7]+constants[8])*ne*this_block_state[483] 
        - (constants[5]*ne+constants[6])*this_block_state[483];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[484] = (constants[5]*ne+constants[6])*this_block_state[483]
        -constants[9]*ne*this_block_state[484]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[485] = constants[2]*ne*this_block_state[486]
        -(constants[0]*ne + constants[1])*this_block_state[485]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[486] = -this_block_derivatives[485];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[487] = (constants[7]+constants[8])*ne*this_block_state[488] 
        - (constants[3]*ne+constants[4])*this_block_state[487];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[488] = constants[9]*ne*this_block_state[489] 
        + (constants[3]*ne+constants[4])*this_block_state[487]  
        - (constants[7]+constants[8])*ne*this_block_state[488] 
        - (constants[5]*ne+constants[6])*this_block_state[488];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[489] = (constants[5]*ne+constants[6])*this_block_state[488]
        -constants[9]*ne*this_block_state[489]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[490] = constants[2]*ne*this_block_state[491]
        -(constants[0]*ne + constants[1])*this_block_state[490]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[491] = -this_block_derivatives[490];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[492] = (constants[7]+constants[8])*ne*this_block_state[493] 
        - (constants[3]*ne+constants[4])*this_block_state[492];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[493] = constants[9]*ne*this_block_state[494] 
        + (constants[3]*ne+constants[4])*this_block_state[492]  
        - (constants[7]+constants[8])*ne*this_block_state[493] 
        - (constants[5]*ne+constants[6])*this_block_state[493];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[494] = (constants[5]*ne+constants[6])*this_block_state[493]
        -constants[9]*ne*this_block_state[494]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[495] = constants[2]*ne*this_block_state[496]
        -(constants[0]*ne + constants[1])*this_block_state[495]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[496] = -this_block_derivatives[495];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[497] = (constants[7]+constants[8])*ne*this_block_state[498] 
        - (constants[3]*ne+constants[4])*this_block_state[497];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[498] = constants[9]*ne*this_block_state[499] 
        + (constants[3]*ne+constants[4])*this_block_state[497]  
        - (constants[7]+constants[8])*ne*this_block_state[498] 
        - (constants[5]*ne+constants[6])*this_block_state[498];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[499] = (constants[5]*ne+constants[6])*this_block_state[498]
        -constants[9]*ne*this_block_state[499]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[500] = constants[2]*ne*this_block_state[501]
        -(constants[0]*ne + constants[1])*this_block_state[500]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[501] = -this_block_derivatives[500];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[502] = (constants[7]+constants[8])*ne*this_block_state[503] 
        - (constants[3]*ne+constants[4])*this_block_state[502];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[503] = constants[9]*ne*this_block_state[504] 
        + (constants[3]*ne+constants[4])*this_block_state[502]  
        - (constants[7]+constants[8])*ne*this_block_state[503] 
        - (constants[5]*ne+constants[6])*this_block_state[503];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[504] = (constants[5]*ne+constants[6])*this_block_state[503]
        -constants[9]*ne*this_block_state[504]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[505] = constants[2]*ne*this_block_state[506]
        -(constants[0]*ne + constants[1])*this_block_state[505]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[506] = -this_block_derivatives[505];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[507] = (constants[7]+constants[8])*ne*this_block_state[508] 
        - (constants[3]*ne+constants[4])*this_block_state[507];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[508] = constants[9]*ne*this_block_state[509] 
        + (constants[3]*ne+constants[4])*this_block_state[507]  
        - (constants[7]+constants[8])*ne*this_block_state[508] 
        - (constants[5]*ne+constants[6])*this_block_state[508];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[509] = (constants[5]*ne+constants[6])*this_block_state[508]
        -constants[9]*ne*this_block_state[509]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[510] = constants[2]*ne*this_block_state[511]
        -(constants[0]*ne + constants[1])*this_block_state[510]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[511] = -this_block_derivatives[510];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[512] = (constants[7]+constants[8])*ne*this_block_state[513] 
        - (constants[3]*ne+constants[4])*this_block_state[512];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[513] = constants[9]*ne*this_block_state[514] 
        + (constants[3]*ne+constants[4])*this_block_state[512]  
        - (constants[7]+constants[8])*ne*this_block_state[513] 
        - (constants[5]*ne+constants[6])*this_block_state[513];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[514] = (constants[5]*ne+constants[6])*this_block_state[513]
        -constants[9]*ne*this_block_state[514]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[515] = constants[2]*ne*this_block_state[516]
        -(constants[0]*ne + constants[1])*this_block_state[515]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[516] = -this_block_derivatives[515];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[517] = (constants[7]+constants[8])*ne*this_block_state[518] 
        - (constants[3]*ne+constants[4])*this_block_state[517];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[518] = constants[9]*ne*this_block_state[519] 
        + (constants[3]*ne+constants[4])*this_block_state[517]  
        - (constants[7]+constants[8])*ne*this_block_state[518] 
        - (constants[5]*ne+constants[6])*this_block_state[518];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[519] = (constants[5]*ne+constants[6])*this_block_state[518]
        -constants[9]*ne*this_block_state[519]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[520] = constants[2]*ne*this_block_state[521]
        -(constants[0]*ne + constants[1])*this_block_state[520]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[521] = -this_block_derivatives[520];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[522] = (constants[7]+constants[8])*ne*this_block_state[523] 
        - (constants[3]*ne+constants[4])*this_block_state[522];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[523] = constants[9]*ne*this_block_state[524] 
        + (constants[3]*ne+constants[4])*this_block_state[522]  
        - (constants[7]+constants[8])*ne*this_block_state[523] 
        - (constants[5]*ne+constants[6])*this_block_state[523];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[524] = (constants[5]*ne+constants[6])*this_block_state[523]
        -constants[9]*ne*this_block_state[524]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[525] = constants[2]*ne*this_block_state[526]
        -(constants[0]*ne + constants[1])*this_block_state[525]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[526] = -this_block_derivatives[525];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[527] = (constants[7]+constants[8])*ne*this_block_state[528] 
        - (constants[3]*ne+constants[4])*this_block_state[527];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[528] = constants[9]*ne*this_block_state[529] 
        + (constants[3]*ne+constants[4])*this_block_state[527]  
        - (constants[7]+constants[8])*ne*this_block_state[528] 
        - (constants[5]*ne+constants[6])*this_block_state[528];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[529] = (constants[5]*ne+constants[6])*this_block_state[528]
        -constants[9]*ne*this_block_state[529]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[530] = constants[2]*ne*this_block_state[531]
        -(constants[0]*ne + constants[1])*this_block_state[530]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[531] = -this_block_derivatives[530];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[532] = (constants[7]+constants[8])*ne*this_block_state[533] 
        - (constants[3]*ne+constants[4])*this_block_state[532];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[533] = constants[9]*ne*this_block_state[534] 
        + (constants[3]*ne+constants[4])*this_block_state[532]  
        - (constants[7]+constants[8])*ne*this_block_state[533] 
        - (constants[5]*ne+constants[6])*this_block_state[533];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[534] = (constants[5]*ne+constants[6])*this_block_state[533]
        -constants[9]*ne*this_block_state[534]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[535] = constants[2]*ne*this_block_state[536]
        -(constants[0]*ne + constants[1])*this_block_state[535]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[536] = -this_block_derivatives[535];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[537] = (constants[7]+constants[8])*ne*this_block_state[538] 
        - (constants[3]*ne+constants[4])*this_block_state[537];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[538] = constants[9]*ne*this_block_state[539] 
        + (constants[3]*ne+constants[4])*this_block_state[537]  
        - (constants[7]+constants[8])*ne*this_block_state[538] 
        - (constants[5]*ne+constants[6])*this_block_state[538];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[539] = (constants[5]*ne+constants[6])*this_block_state[538]
        -constants[9]*ne*this_block_state[539]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[540] = constants[2]*ne*this_block_state[541]
        -(constants[0]*ne + constants[1])*this_block_state[540]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[541] = -this_block_derivatives[540];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[542] = (constants[7]+constants[8])*ne*this_block_state[543] 
        - (constants[3]*ne+constants[4])*this_block_state[542];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[543] = constants[9]*ne*this_block_state[544] 
        + (constants[3]*ne+constants[4])*this_block_state[542]  
        - (constants[7]+constants[8])*ne*this_block_state[543] 
        - (constants[5]*ne+constants[6])*this_block_state[543];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[544] = (constants[5]*ne+constants[6])*this_block_state[543]
        -constants[9]*ne*this_block_state[544]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[545] = constants[2]*ne*this_block_state[546]
        -(constants[0]*ne + constants[1])*this_block_state[545]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[546] = -this_block_derivatives[545];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[547] = (constants[7]+constants[8])*ne*this_block_state[548] 
        - (constants[3]*ne+constants[4])*this_block_state[547];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[548] = constants[9]*ne*this_block_state[549] 
        + (constants[3]*ne+constants[4])*this_block_state[547]  
        - (constants[7]+constants[8])*ne*this_block_state[548] 
        - (constants[5]*ne+constants[6])*this_block_state[548];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[549] = (constants[5]*ne+constants[6])*this_block_state[548]
        -constants[9]*ne*this_block_state[549]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[550] = constants[2]*ne*this_block_state[551]
        -(constants[0]*ne + constants[1])*this_block_state[550]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[551] = -this_block_derivatives[550];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[552] = (constants[7]+constants[8])*ne*this_block_state[553] 
        - (constants[3]*ne+constants[4])*this_block_state[552];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[553] = constants[9]*ne*this_block_state[554] 
        + (constants[3]*ne+constants[4])*this_block_state[552]  
        - (constants[7]+constants[8])*ne*this_block_state[553] 
        - (constants[5]*ne+constants[6])*this_block_state[553];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[554] = (constants[5]*ne+constants[6])*this_block_state[553]
        -constants[9]*ne*this_block_state[554]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[555] = constants[2]*ne*this_block_state[556]
        -(constants[0]*ne + constants[1])*this_block_state[555]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[556] = -this_block_derivatives[555];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[557] = (constants[7]+constants[8])*ne*this_block_state[558] 
        - (constants[3]*ne+constants[4])*this_block_state[557];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[558] = constants[9]*ne*this_block_state[559] 
        + (constants[3]*ne+constants[4])*this_block_state[557]  
        - (constants[7]+constants[8])*ne*this_block_state[558] 
        - (constants[5]*ne+constants[6])*this_block_state[558];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[559] = (constants[5]*ne+constants[6])*this_block_state[558]
        -constants[9]*ne*this_block_state[559]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[560] = constants[2]*ne*this_block_state[561]
        -(constants[0]*ne + constants[1])*this_block_state[560]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[561] = -this_block_derivatives[560];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[562] = (constants[7]+constants[8])*ne*this_block_state[563] 
        - (constants[3]*ne+constants[4])*this_block_state[562];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[563] = constants[9]*ne*this_block_state[564] 
        + (constants[3]*ne+constants[4])*this_block_state[562]  
        - (constants[7]+constants[8])*ne*this_block_state[563] 
        - (constants[5]*ne+constants[6])*this_block_state[563];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[564] = (constants[5]*ne+constants[6])*this_block_state[563]
        -constants[9]*ne*this_block_state[564]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[565] = constants[2]*ne*this_block_state[566]
        -(constants[0]*ne + constants[1])*this_block_state[565]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[566] = -this_block_derivatives[565];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[567] = (constants[7]+constants[8])*ne*this_block_state[568] 
        - (constants[3]*ne+constants[4])*this_block_state[567];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[568] = constants[9]*ne*this_block_state[569] 
        + (constants[3]*ne+constants[4])*this_block_state[567]  
        - (constants[7]+constants[8])*ne*this_block_state[568] 
        - (constants[5]*ne+constants[6])*this_block_state[568];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[569] = (constants[5]*ne+constants[6])*this_block_state[568]
        -constants[9]*ne*this_block_state[569]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[570] = constants[2]*ne*this_block_state[571]
        -(constants[0]*ne + constants[1])*this_block_state[570]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[571] = -this_block_derivatives[570];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[572] = (constants[7]+constants[8])*ne*this_block_state[573] 
        - (constants[3]*ne+constants[4])*this_block_state[572];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[573] = constants[9]*ne*this_block_state[574] 
        + (constants[3]*ne+constants[4])*this_block_state[572]  
        - (constants[7]+constants[8])*ne*this_block_state[573] 
        - (constants[5]*ne+constants[6])*this_block_state[573];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[574] = (constants[5]*ne+constants[6])*this_block_state[573]
        -constants[9]*ne*this_block_state[574]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[575] = constants[2]*ne*this_block_state[576]
        -(constants[0]*ne + constants[1])*this_block_state[575]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[576] = -this_block_derivatives[575];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[577] = (constants[7]+constants[8])*ne*this_block_state[578] 
        - (constants[3]*ne+constants[4])*this_block_state[577];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[578] = constants[9]*ne*this_block_state[579] 
        + (constants[3]*ne+constants[4])*this_block_state[577]  
        - (constants[7]+constants[8])*ne*this_block_state[578] 
        - (constants[5]*ne+constants[6])*this_block_state[578];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[579] = (constants[5]*ne+constants[6])*this_block_state[578]
        -constants[9]*ne*this_block_state[579]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[580] = constants[2]*ne*this_block_state[581]
        -(constants[0]*ne + constants[1])*this_block_state[580]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[581] = -this_block_derivatives[580];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[582] = (constants[7]+constants[8])*ne*this_block_state[583] 
        - (constants[3]*ne+constants[4])*this_block_state[582];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[583] = constants[9]*ne*this_block_state[584] 
        + (constants[3]*ne+constants[4])*this_block_state[582]  
        - (constants[7]+constants[8])*ne*this_block_state[583] 
        - (constants[5]*ne+constants[6])*this_block_state[583];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[584] = (constants[5]*ne+constants[6])*this_block_state[583]
        -constants[9]*ne*this_block_state[584]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[585] = constants[2]*ne*this_block_state[586]
        -(constants[0]*ne + constants[1])*this_block_state[585]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[586] = -this_block_derivatives[585];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[587] = (constants[7]+constants[8])*ne*this_block_state[588] 
        - (constants[3]*ne+constants[4])*this_block_state[587];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[588] = constants[9]*ne*this_block_state[589] 
        + (constants[3]*ne+constants[4])*this_block_state[587]  
        - (constants[7]+constants[8])*ne*this_block_state[588] 
        - (constants[5]*ne+constants[6])*this_block_state[588];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[589] = (constants[5]*ne+constants[6])*this_block_state[588]
        -constants[9]*ne*this_block_state[589]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[590] = constants[2]*ne*this_block_state[591]
        -(constants[0]*ne + constants[1])*this_block_state[590]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[591] = -this_block_derivatives[590];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[592] = (constants[7]+constants[8])*ne*this_block_state[593] 
        - (constants[3]*ne+constants[4])*this_block_state[592];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[593] = constants[9]*ne*this_block_state[594] 
        + (constants[3]*ne+constants[4])*this_block_state[592]  
        - (constants[7]+constants[8])*ne*this_block_state[593] 
        - (constants[5]*ne+constants[6])*this_block_state[593];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[594] = (constants[5]*ne+constants[6])*this_block_state[593]
        -constants[9]*ne*this_block_state[594]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[595] = constants[2]*ne*this_block_state[596]
        -(constants[0]*ne + constants[1])*this_block_state[595]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[596] = -this_block_derivatives[595];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[597] = (constants[7]+constants[8])*ne*this_block_state[598] 
        - (constants[3]*ne+constants[4])*this_block_state[597];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[598] = constants[9]*ne*this_block_state[599] 
        + (constants[3]*ne+constants[4])*this_block_state[597]  
        - (constants[7]+constants[8])*ne*this_block_state[598] 
        - (constants[5]*ne+constants[6])*this_block_state[598];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[599] = (constants[5]*ne+constants[6])*this_block_state[598]
        -constants[9]*ne*this_block_state[599]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[600] = constants[2]*ne*this_block_state[601]
        -(constants[0]*ne + constants[1])*this_block_state[600]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[601] = -this_block_derivatives[600];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[602] = (constants[7]+constants[8])*ne*this_block_state[603] 
        - (constants[3]*ne+constants[4])*this_block_state[602];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[603] = constants[9]*ne*this_block_state[604] 
        + (constants[3]*ne+constants[4])*this_block_state[602]  
        - (constants[7]+constants[8])*ne*this_block_state[603] 
        - (constants[5]*ne+constants[6])*this_block_state[603];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[604] = (constants[5]*ne+constants[6])*this_block_state[603]
        -constants[9]*ne*this_block_state[604]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[605] = constants[2]*ne*this_block_state[606]
        -(constants[0]*ne + constants[1])*this_block_state[605]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[606] = -this_block_derivatives[605];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[607] = (constants[7]+constants[8])*ne*this_block_state[608] 
        - (constants[3]*ne+constants[4])*this_block_state[607];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[608] = constants[9]*ne*this_block_state[609] 
        + (constants[3]*ne+constants[4])*this_block_state[607]  
        - (constants[7]+constants[8])*ne*this_block_state[608] 
        - (constants[5]*ne+constants[6])*this_block_state[608];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[609] = (constants[5]*ne+constants[6])*this_block_state[608]
        -constants[9]*ne*this_block_state[609]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[610] = constants[2]*ne*this_block_state[611]
        -(constants[0]*ne + constants[1])*this_block_state[610]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[611] = -this_block_derivatives[610];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[612] = (constants[7]+constants[8])*ne*this_block_state[613] 
        - (constants[3]*ne+constants[4])*this_block_state[612];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[613] = constants[9]*ne*this_block_state[614] 
        + (constants[3]*ne+constants[4])*this_block_state[612]  
        - (constants[7]+constants[8])*ne*this_block_state[613] 
        - (constants[5]*ne+constants[6])*this_block_state[613];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[614] = (constants[5]*ne+constants[6])*this_block_state[613]
        -constants[9]*ne*this_block_state[614]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[615] = constants[2]*ne*this_block_state[616]
        -(constants[0]*ne + constants[1])*this_block_state[615]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[616] = -this_block_derivatives[615];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[617] = (constants[7]+constants[8])*ne*this_block_state[618] 
        - (constants[3]*ne+constants[4])*this_block_state[617];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[618] = constants[9]*ne*this_block_state[619] 
        + (constants[3]*ne+constants[4])*this_block_state[617]  
        - (constants[7]+constants[8])*ne*this_block_state[618] 
        - (constants[5]*ne+constants[6])*this_block_state[618];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[619] = (constants[5]*ne+constants[6])*this_block_state[618]
        -constants[9]*ne*this_block_state[619]; 
        // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[620] = constants[2]*ne*this_block_state[621]
        -(constants[0]*ne + constants[1])*this_block_state[620]; 
    
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[621] = -this_block_derivatives[620];
    
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[622] = (constants[7]+constants[8])*ne*this_block_state[623] 
        - (constants[3]*ne+constants[4])*this_block_state[622];
    
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[623] = constants[9]*ne*this_block_state[624] 
        + (constants[3]*ne+constants[4])*this_block_state[622]  
        - (constants[7]+constants[8])*ne*this_block_state[623] 
        - (constants[5]*ne+constants[6])*this_block_state[623];
    
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[624] = (constants[5]*ne+constants[6])*this_block_state[623]
        -constants[9]*ne*this_block_state[624]; 
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
    this_block_jacobian[626] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[625] = -this_block_jacobian[626]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[1252] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[1253] = this_block_jacobian[1252]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[1879] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[1877] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[1878] = -this_block_jacobian[1877] - 
        this_block_jacobian[1879]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[2504] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[2503] = -this_block_jacobian[2504];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[3130] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[3131] = -this_block_jacobian[3130]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[3756] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[3755] = -this_block_jacobian[3756]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[4382] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[4383] = this_block_jacobian[4382]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[5009] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[5007] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[5008] = -this_block_jacobian[5007] - 
        this_block_jacobian[5009]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[5634] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[5633] = -this_block_jacobian[5634];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[6260] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[6261] = -this_block_jacobian[6260]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[6886] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[6885] = -this_block_jacobian[6886]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[7512] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[7513] = this_block_jacobian[7512]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[8139] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[8137] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[8138] = -this_block_jacobian[8137] - 
        this_block_jacobian[8139]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[8764] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[8763] = -this_block_jacobian[8764];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[9390] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[9391] = -this_block_jacobian[9390]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[10016] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[10015] = -this_block_jacobian[10016]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[10642] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[10643] = this_block_jacobian[10642]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[11269] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[11267] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[11268] = -this_block_jacobian[11267] - 
        this_block_jacobian[11269]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[11894] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[11893] = -this_block_jacobian[11894];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[12520] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[12521] = -this_block_jacobian[12520]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[13146] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[13145] = -this_block_jacobian[13146]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[13772] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[13773] = this_block_jacobian[13772]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[14399] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[14397] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[14398] = -this_block_jacobian[14397] - 
        this_block_jacobian[14399]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[15024] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[15023] = -this_block_jacobian[15024];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[15650] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[15651] = -this_block_jacobian[15650]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[16276] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[16275] = -this_block_jacobian[16276]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[16902] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[16903] = this_block_jacobian[16902]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[17529] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[17527] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[17528] = -this_block_jacobian[17527] - 
        this_block_jacobian[17529]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[18154] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[18153] = -this_block_jacobian[18154];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[18780] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[18781] = -this_block_jacobian[18780]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[19406] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[19405] = -this_block_jacobian[19406]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[20032] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[20033] = this_block_jacobian[20032]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[20659] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[20657] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[20658] = -this_block_jacobian[20657] - 
        this_block_jacobian[20659]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[21284] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[21283] = -this_block_jacobian[21284];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[21910] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[21911] = -this_block_jacobian[21910]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[22536] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[22535] = -this_block_jacobian[22536]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[23162] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[23163] = this_block_jacobian[23162]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[23789] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[23787] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[23788] = -this_block_jacobian[23787] - 
        this_block_jacobian[23789]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[24414] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[24413] = -this_block_jacobian[24414];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[25040] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[25041] = -this_block_jacobian[25040]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[25666] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[25665] = -this_block_jacobian[25666]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[26292] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[26293] = this_block_jacobian[26292]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[26919] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[26917] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[26918] = -this_block_jacobian[26917] - 
        this_block_jacobian[26919]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[27544] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[27543] = -this_block_jacobian[27544];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[28170] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[28171] = -this_block_jacobian[28170]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[28796] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[28795] = -this_block_jacobian[28796]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[29422] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[29423] = this_block_jacobian[29422]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[30049] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[30047] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[30048] = -this_block_jacobian[30047] - 
        this_block_jacobian[30049]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[30674] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[30673] = -this_block_jacobian[30674];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[31300] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[31301] = -this_block_jacobian[31300]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[31926] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[31925] = -this_block_jacobian[31926]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[32552] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[32553] = this_block_jacobian[32552]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[33179] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[33177] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[33178] = -this_block_jacobian[33177] - 
        this_block_jacobian[33179]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[33804] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[33803] = -this_block_jacobian[33804];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[34430] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[34431] = -this_block_jacobian[34430]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[35056] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[35055] = -this_block_jacobian[35056]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[35682] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[35683] = this_block_jacobian[35682]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[36309] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[36307] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[36308] = -this_block_jacobian[36307] - 
        this_block_jacobian[36309]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[36934] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[36933] = -this_block_jacobian[36934];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[37560] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[37561] = -this_block_jacobian[37560]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[38186] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[38185] = -this_block_jacobian[38186]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[38812] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[38813] = this_block_jacobian[38812]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[39439] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[39437] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[39438] = -this_block_jacobian[39437] - 
        this_block_jacobian[39439]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[40064] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[40063] = -this_block_jacobian[40064];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[40690] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[40691] = -this_block_jacobian[40690]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[41316] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[41315] = -this_block_jacobian[41316]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[41942] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[41943] = this_block_jacobian[41942]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[42569] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[42567] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[42568] = -this_block_jacobian[42567] - 
        this_block_jacobian[42569]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[43194] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[43193] = -this_block_jacobian[43194];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[43820] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[43821] = -this_block_jacobian[43820]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[44446] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[44445] = -this_block_jacobian[44446]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[45072] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[45073] = this_block_jacobian[45072]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[45699] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[45697] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[45698] = -this_block_jacobian[45697] - 
        this_block_jacobian[45699]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[46324] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[46323] = -this_block_jacobian[46324];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[46950] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[46951] = -this_block_jacobian[46950]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[47576] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[47575] = -this_block_jacobian[47576]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[48202] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[48203] = this_block_jacobian[48202]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[48829] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[48827] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[48828] = -this_block_jacobian[48827] - 
        this_block_jacobian[48829]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[49454] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[49453] = -this_block_jacobian[49454];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[50080] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[50081] = -this_block_jacobian[50080]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[50706] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[50705] = -this_block_jacobian[50706]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[51332] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[51333] = this_block_jacobian[51332]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[51959] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[51957] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[51958] = -this_block_jacobian[51957] - 
        this_block_jacobian[51959]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[52584] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[52583] = -this_block_jacobian[52584];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[53210] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[53211] = -this_block_jacobian[53210]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[53836] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[53835] = -this_block_jacobian[53836]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[54462] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[54463] = this_block_jacobian[54462]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[55089] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[55087] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[55088] = -this_block_jacobian[55087] - 
        this_block_jacobian[55089]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[55714] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[55713] = -this_block_jacobian[55714];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[56340] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[56341] = -this_block_jacobian[56340]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[56966] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[56965] = -this_block_jacobian[56966]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[57592] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[57593] = this_block_jacobian[57592]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[58219] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[58217] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[58218] = -this_block_jacobian[58217] - 
        this_block_jacobian[58219]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[58844] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[58843] = -this_block_jacobian[58844];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[59470] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[59471] = -this_block_jacobian[59470]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[60096] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[60095] = -this_block_jacobian[60096]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[60722] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[60723] = this_block_jacobian[60722]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[61349] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[61347] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[61348] = -this_block_jacobian[61347] - 
        this_block_jacobian[61349]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[61974] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[61973] = -this_block_jacobian[61974];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[62600] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[62601] = -this_block_jacobian[62600]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[63226] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[63225] = -this_block_jacobian[63226]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[63852] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[63853] = this_block_jacobian[63852]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[64479] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[64477] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[64478] = -this_block_jacobian[64477] - 
        this_block_jacobian[64479]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[65104] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[65103] = -this_block_jacobian[65104];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[65730] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[65731] = -this_block_jacobian[65730]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[66356] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[66355] = -this_block_jacobian[66356]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[66982] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[66983] = this_block_jacobian[66982]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[67609] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[67607] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[67608] = -this_block_jacobian[67607] - 
        this_block_jacobian[67609]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[68234] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[68233] = -this_block_jacobian[68234];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[68860] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[68861] = -this_block_jacobian[68860]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[69486] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[69485] = -this_block_jacobian[69486]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[70112] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[70113] = this_block_jacobian[70112]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[70739] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[70737] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[70738] = -this_block_jacobian[70737] - 
        this_block_jacobian[70739]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[71364] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[71363] = -this_block_jacobian[71364];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[71990] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[71991] = -this_block_jacobian[71990]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[72616] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[72615] = -this_block_jacobian[72616]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[73242] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[73243] = this_block_jacobian[73242]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[73869] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[73867] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[73868] = -this_block_jacobian[73867] - 
        this_block_jacobian[73869]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[74494] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[74493] = -this_block_jacobian[74494];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[75120] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[75121] = -this_block_jacobian[75120]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[75746] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[75745] = -this_block_jacobian[75746]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[76372] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[76373] = this_block_jacobian[76372]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[76999] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[76997] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[76998] = -this_block_jacobian[76997] - 
        this_block_jacobian[76999]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[77624] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[77623] = -this_block_jacobian[77624];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[78250] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[78251] = -this_block_jacobian[78250]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[78876] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[78875] = -this_block_jacobian[78876]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[79502] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[79503] = this_block_jacobian[79502]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[80129] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[80127] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[80128] = -this_block_jacobian[80127] - 
        this_block_jacobian[80129]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[80754] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[80753] = -this_block_jacobian[80754];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[81380] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[81381] = -this_block_jacobian[81380]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[82006] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[82005] = -this_block_jacobian[82006]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[82632] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[82633] = this_block_jacobian[82632]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[83259] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[83257] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[83258] = -this_block_jacobian[83257] - 
        this_block_jacobian[83259]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[83884] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[83883] = -this_block_jacobian[83884];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[84510] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[84511] = -this_block_jacobian[84510]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[85136] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[85135] = -this_block_jacobian[85136]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[85762] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[85763] = this_block_jacobian[85762]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[86389] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[86387] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[86388] = -this_block_jacobian[86387] - 
        this_block_jacobian[86389]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[87014] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[87013] = -this_block_jacobian[87014];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[87640] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[87641] = -this_block_jacobian[87640]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[88266] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[88265] = -this_block_jacobian[88266]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[88892] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[88893] = this_block_jacobian[88892]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[89519] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[89517] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[89518] = -this_block_jacobian[89517] - 
        this_block_jacobian[89519]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[90144] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[90143] = -this_block_jacobian[90144];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[90770] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[90771] = -this_block_jacobian[90770]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[91396] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[91395] = -this_block_jacobian[91396]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[92022] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[92023] = this_block_jacobian[92022]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[92649] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[92647] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[92648] = -this_block_jacobian[92647] - 
        this_block_jacobian[92649]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[93274] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[93273] = -this_block_jacobian[93274];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[93900] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[93901] = -this_block_jacobian[93900]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[94526] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[94525] = -this_block_jacobian[94526]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[95152] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[95153] = this_block_jacobian[95152]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[95779] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[95777] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[95778] = -this_block_jacobian[95777] - 
        this_block_jacobian[95779]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[96404] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[96403] = -this_block_jacobian[96404];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[97030] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[97031] = -this_block_jacobian[97030]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[97656] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[97655] = -this_block_jacobian[97656]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[98282] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[98283] = this_block_jacobian[98282]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[98909] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[98907] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[98908] = -this_block_jacobian[98907] - 
        this_block_jacobian[98909]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[99534] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[99533] = -this_block_jacobian[99534];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[100160] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[100161] = -this_block_jacobian[100160]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[100786] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[100785] = -this_block_jacobian[100786]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[101412] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[101413] = this_block_jacobian[101412]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[102039] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[102037] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[102038] = -this_block_jacobian[102037] - 
        this_block_jacobian[102039]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[102664] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[102663] = -this_block_jacobian[102664];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[103290] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[103291] = -this_block_jacobian[103290]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[103916] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[103915] = -this_block_jacobian[103916]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[104542] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[104543] = this_block_jacobian[104542]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[105169] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[105167] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[105168] = -this_block_jacobian[105167] - 
        this_block_jacobian[105169]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[105794] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[105793] = -this_block_jacobian[105794];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[106420] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[106421] = -this_block_jacobian[106420]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[107046] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[107045] = -this_block_jacobian[107046]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[107672] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[107673] = this_block_jacobian[107672]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[108299] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[108297] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[108298] = -this_block_jacobian[108297] - 
        this_block_jacobian[108299]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[108924] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[108923] = -this_block_jacobian[108924];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[109550] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[109551] = -this_block_jacobian[109550]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[110176] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[110175] = -this_block_jacobian[110176]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[110802] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[110803] = this_block_jacobian[110802]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[111429] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[111427] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[111428] = -this_block_jacobian[111427] - 
        this_block_jacobian[111429]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[112054] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[112053] = -this_block_jacobian[112054];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[112680] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[112681] = -this_block_jacobian[112680]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[113306] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[113305] = -this_block_jacobian[113306]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[113932] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[113933] = this_block_jacobian[113932]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[114559] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[114557] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[114558] = -this_block_jacobian[114557] - 
        this_block_jacobian[114559]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[115184] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[115183] = -this_block_jacobian[115184];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[115810] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[115811] = -this_block_jacobian[115810]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[116436] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[116435] = -this_block_jacobian[116436]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[117062] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[117063] = this_block_jacobian[117062]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[117689] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[117687] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[117688] = -this_block_jacobian[117687] - 
        this_block_jacobian[117689]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[118314] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[118313] = -this_block_jacobian[118314];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[118940] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[118941] = -this_block_jacobian[118940]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[119566] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[119565] = -this_block_jacobian[119566]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[120192] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[120193] = this_block_jacobian[120192]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[120819] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[120817] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[120818] = -this_block_jacobian[120817] - 
        this_block_jacobian[120819]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[121444] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[121443] = -this_block_jacobian[121444];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[122070] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[122071] = -this_block_jacobian[122070]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[122696] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[122695] = -this_block_jacobian[122696]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[123322] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[123323] = this_block_jacobian[123322]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[123949] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[123947] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[123948] = -this_block_jacobian[123947] - 
        this_block_jacobian[123949]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[124574] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[124573] = -this_block_jacobian[124574];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[125200] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[125201] = -this_block_jacobian[125200]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[125826] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[125825] = -this_block_jacobian[125826]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[126452] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[126453] = this_block_jacobian[126452]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[127079] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[127077] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[127078] = -this_block_jacobian[127077] - 
        this_block_jacobian[127079]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[127704] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[127703] = -this_block_jacobian[127704];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[128330] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[128331] = -this_block_jacobian[128330]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[128956] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[128955] = -this_block_jacobian[128956]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[129582] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[129583] = this_block_jacobian[129582]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[130209] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[130207] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[130208] = -this_block_jacobian[130207] - 
        this_block_jacobian[130209]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[130834] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[130833] = -this_block_jacobian[130834];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[131460] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[131461] = -this_block_jacobian[131460]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[132086] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[132085] = -this_block_jacobian[132086]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[132712] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[132713] = this_block_jacobian[132712]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[133339] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[133337] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[133338] = -this_block_jacobian[133337] - 
        this_block_jacobian[133339]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[133964] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[133963] = -this_block_jacobian[133964];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[134590] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[134591] = -this_block_jacobian[134590]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[135216] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[135215] = -this_block_jacobian[135216]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[135842] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[135843] = this_block_jacobian[135842]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[136469] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[136467] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[136468] = -this_block_jacobian[136467] - 
        this_block_jacobian[136469]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[137094] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[137093] = -this_block_jacobian[137094];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[137720] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[137721] = -this_block_jacobian[137720]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[138346] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[138345] = -this_block_jacobian[138346]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[138972] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[138973] = this_block_jacobian[138972]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[139599] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[139597] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[139598] = -this_block_jacobian[139597] - 
        this_block_jacobian[139599]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[140224] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[140223] = -this_block_jacobian[140224];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[140850] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[140851] = -this_block_jacobian[140850]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[141476] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[141475] = -this_block_jacobian[141476]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[142102] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[142103] = this_block_jacobian[142102]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[142729] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[142727] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[142728] = -this_block_jacobian[142727] - 
        this_block_jacobian[142729]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[143354] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[143353] = -this_block_jacobian[143354];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[143980] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[143981] = -this_block_jacobian[143980]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[144606] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[144605] = -this_block_jacobian[144606]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[145232] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[145233] = this_block_jacobian[145232]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[145859] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[145857] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[145858] = -this_block_jacobian[145857] - 
        this_block_jacobian[145859]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[146484] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[146483] = -this_block_jacobian[146484];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[147110] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[147111] = -this_block_jacobian[147110]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[147736] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[147735] = -this_block_jacobian[147736]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[148362] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[148363] = this_block_jacobian[148362]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[148989] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[148987] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[148988] = -this_block_jacobian[148987] - 
        this_block_jacobian[148989]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[149614] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[149613] = -this_block_jacobian[149614];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[150240] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[150241] = -this_block_jacobian[150240]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[150866] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[150865] = -this_block_jacobian[150866]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[151492] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[151493] = this_block_jacobian[151492]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[152119] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[152117] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[152118] = -this_block_jacobian[152117] - 
        this_block_jacobian[152119]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[152744] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[152743] = -this_block_jacobian[152744];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[153370] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[153371] = -this_block_jacobian[153370]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[153996] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[153995] = -this_block_jacobian[153996]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[154622] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[154623] = this_block_jacobian[154622]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[155249] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[155247] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[155248] = -this_block_jacobian[155247] - 
        this_block_jacobian[155249]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[155874] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[155873] = -this_block_jacobian[155874];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[156500] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[156501] = -this_block_jacobian[156500]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[157126] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[157125] = -this_block_jacobian[157126]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[157752] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[157753] = this_block_jacobian[157752]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[158379] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[158377] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[158378] = -this_block_jacobian[158377] - 
        this_block_jacobian[158379]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[159004] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[159003] = -this_block_jacobian[159004];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[159630] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[159631] = -this_block_jacobian[159630]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[160256] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[160255] = -this_block_jacobian[160256]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[160882] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[160883] = this_block_jacobian[160882]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[161509] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[161507] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[161508] = -this_block_jacobian[161507] - 
        this_block_jacobian[161509]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[162134] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[162133] = -this_block_jacobian[162134];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[162760] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[162761] = -this_block_jacobian[162760]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[163386] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[163385] = -this_block_jacobian[163386]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[164012] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[164013] = this_block_jacobian[164012]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[164639] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[164637] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[164638] = -this_block_jacobian[164637] - 
        this_block_jacobian[164639]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[165264] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[165263] = -this_block_jacobian[165264];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[165890] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[165891] = -this_block_jacobian[165890]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[166516] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[166515] = -this_block_jacobian[166516]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[167142] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[167143] = this_block_jacobian[167142]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[167769] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[167767] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[167768] = -this_block_jacobian[167767] - 
        this_block_jacobian[167769]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[168394] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[168393] = -this_block_jacobian[168394];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[169020] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[169021] = -this_block_jacobian[169020]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[169646] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[169645] = -this_block_jacobian[169646]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[170272] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[170273] = this_block_jacobian[170272]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[170899] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[170897] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[170898] = -this_block_jacobian[170897] - 
        this_block_jacobian[170899]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[171524] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[171523] = -this_block_jacobian[171524];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[172150] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[172151] = -this_block_jacobian[172150]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[172776] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[172775] = -this_block_jacobian[172776]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[173402] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[173403] = this_block_jacobian[173402]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[174029] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[174027] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[174028] = -this_block_jacobian[174027] - 
        this_block_jacobian[174029]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[174654] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[174653] = -this_block_jacobian[174654];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[175280] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[175281] = -this_block_jacobian[175280]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[175906] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[175905] = -this_block_jacobian[175906]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[176532] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[176533] = this_block_jacobian[176532]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[177159] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[177157] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[177158] = -this_block_jacobian[177157] - 
        this_block_jacobian[177159]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[177784] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[177783] = -this_block_jacobian[177784];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[178410] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[178411] = -this_block_jacobian[178410]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[179036] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[179035] = -this_block_jacobian[179036]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[179662] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[179663] = this_block_jacobian[179662]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[180289] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[180287] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[180288] = -this_block_jacobian[180287] - 
        this_block_jacobian[180289]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[180914] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[180913] = -this_block_jacobian[180914];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[181540] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[181541] = -this_block_jacobian[181540]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[182166] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[182165] = -this_block_jacobian[182166]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[182792] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[182793] = this_block_jacobian[182792]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[183419] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[183417] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[183418] = -this_block_jacobian[183417] - 
        this_block_jacobian[183419]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[184044] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[184043] = -this_block_jacobian[184044];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[184670] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[184671] = -this_block_jacobian[184670]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[185296] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[185295] = -this_block_jacobian[185296]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[185922] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[185923] = this_block_jacobian[185922]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[186549] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[186547] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[186548] = -this_block_jacobian[186547] - 
        this_block_jacobian[186549]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[187174] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[187173] = -this_block_jacobian[187174];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[187800] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[187801] = -this_block_jacobian[187800]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[188426] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[188425] = -this_block_jacobian[188426]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[189052] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[189053] = this_block_jacobian[189052]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[189679] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[189677] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[189678] = -this_block_jacobian[189677] - 
        this_block_jacobian[189679]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[190304] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[190303] = -this_block_jacobian[190304];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[190930] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[190931] = -this_block_jacobian[190930]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[191556] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[191555] = -this_block_jacobian[191556]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[192182] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[192183] = this_block_jacobian[192182]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[192809] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[192807] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[192808] = -this_block_jacobian[192807] - 
        this_block_jacobian[192809]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[193434] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[193433] = -this_block_jacobian[193434];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[194060] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[194061] = -this_block_jacobian[194060]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[194686] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[194685] = -this_block_jacobian[194686]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[195312] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[195313] = this_block_jacobian[195312]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[195939] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[195937] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[195938] = -this_block_jacobian[195937] - 
        this_block_jacobian[195939]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[196564] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[196563] = -this_block_jacobian[196564];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[197190] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[197191] = -this_block_jacobian[197190]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[197816] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[197815] = -this_block_jacobian[197816]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[198442] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[198443] = this_block_jacobian[198442]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[199069] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[199067] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[199068] = -this_block_jacobian[199067] - 
        this_block_jacobian[199069]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[199694] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[199693] = -this_block_jacobian[199694];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[200320] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[200321] = -this_block_jacobian[200320]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[200946] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[200945] = -this_block_jacobian[200946]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[201572] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[201573] = this_block_jacobian[201572]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[202199] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[202197] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[202198] = -this_block_jacobian[202197] - 
        this_block_jacobian[202199]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[202824] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[202823] = -this_block_jacobian[202824];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[203450] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[203451] = -this_block_jacobian[203450]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[204076] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[204075] = -this_block_jacobian[204076]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[204702] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[204703] = this_block_jacobian[204702]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[205329] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[205327] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[205328] = -this_block_jacobian[205327] - 
        this_block_jacobian[205329]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[205954] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[205953] = -this_block_jacobian[205954];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[206580] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[206581] = -this_block_jacobian[206580]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[207206] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[207205] = -this_block_jacobian[207206]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[207832] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[207833] = this_block_jacobian[207832]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[208459] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[208457] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[208458] = -this_block_jacobian[208457] - 
        this_block_jacobian[208459]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[209084] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[209083] = -this_block_jacobian[209084];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[209710] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[209711] = -this_block_jacobian[209710]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[210336] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[210335] = -this_block_jacobian[210336]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[210962] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[210963] = this_block_jacobian[210962]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[211589] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[211587] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[211588] = -this_block_jacobian[211587] - 
        this_block_jacobian[211589]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[212214] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[212213] = -this_block_jacobian[212214];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[212840] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[212841] = -this_block_jacobian[212840]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[213466] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[213465] = -this_block_jacobian[213466]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[214092] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[214093] = this_block_jacobian[214092]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[214719] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[214717] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[214718] = -this_block_jacobian[214717] - 
        this_block_jacobian[214719]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[215344] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[215343] = -this_block_jacobian[215344];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[215970] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[215971] = -this_block_jacobian[215970]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[216596] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[216595] = -this_block_jacobian[216596]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[217222] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[217223] = this_block_jacobian[217222]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[217849] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[217847] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[217848] = -this_block_jacobian[217847] - 
        this_block_jacobian[217849]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[218474] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[218473] = -this_block_jacobian[218474];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[219100] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[219101] = -this_block_jacobian[219100]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[219726] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[219725] = -this_block_jacobian[219726]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[220352] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[220353] = this_block_jacobian[220352]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[220979] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[220977] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[220978] = -this_block_jacobian[220977] - 
        this_block_jacobian[220979]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[221604] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[221603] = -this_block_jacobian[221604];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[222230] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[222231] = -this_block_jacobian[222230]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[222856] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[222855] = -this_block_jacobian[222856]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[223482] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[223483] = this_block_jacobian[223482]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[224109] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[224107] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[224108] = -this_block_jacobian[224107] - 
        this_block_jacobian[224109]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[224734] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[224733] = -this_block_jacobian[224734];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[225360] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[225361] = -this_block_jacobian[225360]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[225986] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[225985] = -this_block_jacobian[225986]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[226612] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[226613] = this_block_jacobian[226612]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[227239] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[227237] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[227238] = -this_block_jacobian[227237] - 
        this_block_jacobian[227239]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[227864] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[227863] = -this_block_jacobian[227864];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[228490] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[228491] = -this_block_jacobian[228490]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[229116] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[229115] = -this_block_jacobian[229116]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[229742] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[229743] = this_block_jacobian[229742]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[230369] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[230367] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[230368] = -this_block_jacobian[230367] - 
        this_block_jacobian[230369]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[230994] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[230993] = -this_block_jacobian[230994];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[231620] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[231621] = -this_block_jacobian[231620]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[232246] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[232245] = -this_block_jacobian[232246]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[232872] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[232873] = this_block_jacobian[232872]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[233499] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[233497] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[233498] = -this_block_jacobian[233497] - 
        this_block_jacobian[233499]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[234124] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[234123] = -this_block_jacobian[234124];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[234750] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[234751] = -this_block_jacobian[234750]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[235376] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[235375] = -this_block_jacobian[235376]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[236002] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[236003] = this_block_jacobian[236002]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[236629] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[236627] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[236628] = -this_block_jacobian[236627] - 
        this_block_jacobian[236629]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[237254] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[237253] = -this_block_jacobian[237254];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[237880] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[237881] = -this_block_jacobian[237880]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[238506] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[238505] = -this_block_jacobian[238506]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[239132] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[239133] = this_block_jacobian[239132]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[239759] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[239757] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[239758] = -this_block_jacobian[239757] - 
        this_block_jacobian[239759]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[240384] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[240383] = -this_block_jacobian[240384];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[241010] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[241011] = -this_block_jacobian[241010]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[241636] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[241635] = -this_block_jacobian[241636]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[242262] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[242263] = this_block_jacobian[242262]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[242889] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[242887] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[242888] = -this_block_jacobian[242887] - 
        this_block_jacobian[242889]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[243514] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[243513] = -this_block_jacobian[243514];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[244140] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[244141] = -this_block_jacobian[244140]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[244766] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[244765] = -this_block_jacobian[244766]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[245392] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[245393] = this_block_jacobian[245392]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[246019] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[246017] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[246018] = -this_block_jacobian[246017] - 
        this_block_jacobian[246019]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[246644] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[246643] = -this_block_jacobian[246644];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[247270] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[247271] = -this_block_jacobian[247270]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[247896] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[247895] = -this_block_jacobian[247896]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[248522] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[248523] = this_block_jacobian[248522]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[249149] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[249147] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[249148] = -this_block_jacobian[249147] - 
        this_block_jacobian[249149]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[249774] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[249773] = -this_block_jacobian[249774];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[250400] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[250401] = -this_block_jacobian[250400]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[251026] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[251025] = -this_block_jacobian[251026]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[251652] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[251653] = this_block_jacobian[251652]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[252279] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[252277] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[252278] = -this_block_jacobian[252277] - 
        this_block_jacobian[252279]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[252904] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[252903] = -this_block_jacobian[252904];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[253530] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[253531] = -this_block_jacobian[253530]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[254156] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[254155] = -this_block_jacobian[254156]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[254782] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[254783] = this_block_jacobian[254782]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[255409] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[255407] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[255408] = -this_block_jacobian[255407] - 
        this_block_jacobian[255409]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[256034] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[256033] = -this_block_jacobian[256034];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[256660] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[256661] = -this_block_jacobian[256660]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[257286] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[257285] = -this_block_jacobian[257286]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[257912] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[257913] = this_block_jacobian[257912]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[258539] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[258537] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[258538] = -this_block_jacobian[258537] - 
        this_block_jacobian[258539]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[259164] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[259163] = -this_block_jacobian[259164];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[259790] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[259791] = -this_block_jacobian[259790]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[260416] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[260415] = -this_block_jacobian[260416]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[261042] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[261043] = this_block_jacobian[261042]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[261669] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[261667] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[261668] = -this_block_jacobian[261667] - 
        this_block_jacobian[261669]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[262294] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[262293] = -this_block_jacobian[262294];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[262920] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[262921] = -this_block_jacobian[262920]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[263546] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[263545] = -this_block_jacobian[263546]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[264172] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[264173] = this_block_jacobian[264172]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[264799] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[264797] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[264798] = -this_block_jacobian[264797] - 
        this_block_jacobian[264799]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[265424] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[265423] = -this_block_jacobian[265424];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[266050] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[266051] = -this_block_jacobian[266050]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[266676] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[266675] = -this_block_jacobian[266676]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[267302] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[267303] = this_block_jacobian[267302]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[267929] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[267927] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[267928] = -this_block_jacobian[267927] - 
        this_block_jacobian[267929]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[268554] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[268553] = -this_block_jacobian[268554];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[269180] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[269181] = -this_block_jacobian[269180]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[269806] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[269805] = -this_block_jacobian[269806]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[270432] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[270433] = this_block_jacobian[270432]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[271059] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[271057] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[271058] = -this_block_jacobian[271057] - 
        this_block_jacobian[271059]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[271684] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[271683] = -this_block_jacobian[271684];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[272310] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[272311] = -this_block_jacobian[272310]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[272936] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[272935] = -this_block_jacobian[272936]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[273562] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[273563] = this_block_jacobian[273562]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[274189] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[274187] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[274188] = -this_block_jacobian[274187] - 
        this_block_jacobian[274189]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[274814] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[274813] = -this_block_jacobian[274814];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[275440] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[275441] = -this_block_jacobian[275440]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[276066] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[276065] = -this_block_jacobian[276066]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[276692] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[276693] = this_block_jacobian[276692]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[277319] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[277317] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[277318] = -this_block_jacobian[277317] - 
        this_block_jacobian[277319]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[277944] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[277943] = -this_block_jacobian[277944];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[278570] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[278571] = -this_block_jacobian[278570]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[279196] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[279195] = -this_block_jacobian[279196]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[279822] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[279823] = this_block_jacobian[279822]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[280449] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[280447] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[280448] = -this_block_jacobian[280447] - 
        this_block_jacobian[280449]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[281074] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[281073] = -this_block_jacobian[281074];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[281700] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[281701] = -this_block_jacobian[281700]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[282326] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[282325] = -this_block_jacobian[282326]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[282952] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[282953] = this_block_jacobian[282952]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[283579] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[283577] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[283578] = -this_block_jacobian[283577] - 
        this_block_jacobian[283579]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[284204] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[284203] = -this_block_jacobian[284204];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[284830] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[284831] = -this_block_jacobian[284830]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[285456] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[285455] = -this_block_jacobian[285456]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[286082] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[286083] = this_block_jacobian[286082]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[286709] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[286707] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[286708] = -this_block_jacobian[286707] - 
        this_block_jacobian[286709]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[287334] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[287333] = -this_block_jacobian[287334];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[287960] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[287961] = -this_block_jacobian[287960]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[288586] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[288585] = -this_block_jacobian[288586]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[289212] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[289213] = this_block_jacobian[289212]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[289839] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[289837] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[289838] = -this_block_jacobian[289837] - 
        this_block_jacobian[289839]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[290464] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[290463] = -this_block_jacobian[290464];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[291090] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[291091] = -this_block_jacobian[291090]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[291716] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[291715] = -this_block_jacobian[291716]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[292342] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[292343] = this_block_jacobian[292342]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[292969] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[292967] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[292968] = -this_block_jacobian[292967] - 
        this_block_jacobian[292969]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[293594] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[293593] = -this_block_jacobian[293594];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[294220] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[294221] = -this_block_jacobian[294220]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[294846] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[294845] = -this_block_jacobian[294846]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[295472] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[295473] = this_block_jacobian[295472]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[296099] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[296097] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[296098] = -this_block_jacobian[296097] - 
        this_block_jacobian[296099]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[296724] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[296723] = -this_block_jacobian[296724];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[297350] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[297351] = -this_block_jacobian[297350]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[297976] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[297975] = -this_block_jacobian[297976]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[298602] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[298603] = this_block_jacobian[298602]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[299229] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[299227] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[299228] = -this_block_jacobian[299227] - 
        this_block_jacobian[299229]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[299854] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[299853] = -this_block_jacobian[299854];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[300480] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[300481] = -this_block_jacobian[300480]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[301106] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[301105] = -this_block_jacobian[301106]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[301732] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[301733] = this_block_jacobian[301732]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[302359] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[302357] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[302358] = -this_block_jacobian[302357] - 
        this_block_jacobian[302359]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[302984] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[302983] = -this_block_jacobian[302984];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[303610] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[303611] = -this_block_jacobian[303610]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[304236] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[304235] = -this_block_jacobian[304236]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[304862] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[304863] = this_block_jacobian[304862]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[305489] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[305487] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[305488] = -this_block_jacobian[305487] - 
        this_block_jacobian[305489]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[306114] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[306113] = -this_block_jacobian[306114];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[306740] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[306741] = -this_block_jacobian[306740]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[307366] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[307365] = -this_block_jacobian[307366]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[307992] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[307993] = this_block_jacobian[307992]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[308619] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[308617] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[308618] = -this_block_jacobian[308617] - 
        this_block_jacobian[308619]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[309244] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[309243] = -this_block_jacobian[309244];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[309870] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[309871] = -this_block_jacobian[309870]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[310496] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[310495] = -this_block_jacobian[310496]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[311122] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[311123] = this_block_jacobian[311122]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[311749] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[311747] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[311748] = -this_block_jacobian[311747] - 
        this_block_jacobian[311749]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[312374] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[312373] = -this_block_jacobian[312374];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[313000] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[313001] = -this_block_jacobian[313000]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[313626] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[313625] = -this_block_jacobian[313626]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[314252] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[314253] = this_block_jacobian[314252]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[314879] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[314877] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[314878] = -this_block_jacobian[314877] - 
        this_block_jacobian[314879]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[315504] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[315503] = -this_block_jacobian[315504];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[316130] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[316131] = -this_block_jacobian[316130]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[316756] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[316755] = -this_block_jacobian[316756]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[317382] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[317383] = this_block_jacobian[317382]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[318009] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[318007] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[318008] = -this_block_jacobian[318007] - 
        this_block_jacobian[318009]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[318634] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[318633] = -this_block_jacobian[318634];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[319260] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[319261] = -this_block_jacobian[319260]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[319886] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[319885] = -this_block_jacobian[319886]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[320512] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[320513] = this_block_jacobian[320512]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[321139] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[321137] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[321138] = -this_block_jacobian[321137] - 
        this_block_jacobian[321139]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[321764] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[321763] = -this_block_jacobian[321764];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[322390] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[322391] = -this_block_jacobian[322390]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[323016] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[323015] = -this_block_jacobian[323016]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[323642] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[323643] = this_block_jacobian[323642]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[324269] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[324267] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[324268] = -this_block_jacobian[324267] - 
        this_block_jacobian[324269]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[324894] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[324893] = -this_block_jacobian[324894];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[325520] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[325521] = -this_block_jacobian[325520]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[326146] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[326145] = -this_block_jacobian[326146]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[326772] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[326773] = this_block_jacobian[326772]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[327399] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[327397] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[327398] = -this_block_jacobian[327397] - 
        this_block_jacobian[327399]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[328024] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[328023] = -this_block_jacobian[328024];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[328650] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[328651] = -this_block_jacobian[328650]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[329276] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[329275] = -this_block_jacobian[329276]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[329902] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[329903] = this_block_jacobian[329902]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[330529] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[330527] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[330528] = -this_block_jacobian[330527] - 
        this_block_jacobian[330529]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[331154] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[331153] = -this_block_jacobian[331154];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[331780] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[331781] = -this_block_jacobian[331780]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[332406] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[332405] = -this_block_jacobian[332406]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[333032] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[333033] = this_block_jacobian[333032]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[333659] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[333657] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[333658] = -this_block_jacobian[333657] - 
        this_block_jacobian[333659]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[334284] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[334283] = -this_block_jacobian[334284];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[334910] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[334911] = -this_block_jacobian[334910]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[335536] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[335535] = -this_block_jacobian[335536]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[336162] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[336163] = this_block_jacobian[336162]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[336789] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[336787] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[336788] = -this_block_jacobian[336787] - 
        this_block_jacobian[336789]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[337414] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[337413] = -this_block_jacobian[337414];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[338040] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[338041] = -this_block_jacobian[338040]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[338666] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[338665] = -this_block_jacobian[338666]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[339292] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[339293] = this_block_jacobian[339292]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[339919] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[339917] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[339918] = -this_block_jacobian[339917] - 
        this_block_jacobian[339919]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[340544] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[340543] = -this_block_jacobian[340544];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[341170] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[341171] = -this_block_jacobian[341170]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[341796] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[341795] = -this_block_jacobian[341796]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[342422] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[342423] = this_block_jacobian[342422]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[343049] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[343047] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[343048] = -this_block_jacobian[343047] - 
        this_block_jacobian[343049]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[343674] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[343673] = -this_block_jacobian[343674];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[344300] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[344301] = -this_block_jacobian[344300]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[344926] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[344925] = -this_block_jacobian[344926]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[345552] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[345553] = this_block_jacobian[345552]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[346179] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[346177] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[346178] = -this_block_jacobian[346177] - 
        this_block_jacobian[346179]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[346804] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[346803] = -this_block_jacobian[346804];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[347430] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[347431] = -this_block_jacobian[347430]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[348056] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[348055] = -this_block_jacobian[348056]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[348682] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[348683] = this_block_jacobian[348682]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[349309] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[349307] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[349308] = -this_block_jacobian[349307] - 
        this_block_jacobian[349309]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[349934] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[349933] = -this_block_jacobian[349934];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[350560] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[350561] = -this_block_jacobian[350560]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[351186] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[351185] = -this_block_jacobian[351186]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[351812] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[351813] = this_block_jacobian[351812]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[352439] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[352437] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[352438] = -this_block_jacobian[352437] - 
        this_block_jacobian[352439]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[353064] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[353063] = -this_block_jacobian[353064];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[353690] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[353691] = -this_block_jacobian[353690]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[354316] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[354315] = -this_block_jacobian[354316]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[354942] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[354943] = this_block_jacobian[354942]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[355569] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[355567] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[355568] = -this_block_jacobian[355567] - 
        this_block_jacobian[355569]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[356194] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[356193] = -this_block_jacobian[356194];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[356820] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[356821] = -this_block_jacobian[356820]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[357446] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[357445] = -this_block_jacobian[357446]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[358072] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[358073] = this_block_jacobian[358072]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[358699] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[358697] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[358698] = -this_block_jacobian[358697] - 
        this_block_jacobian[358699]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[359324] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[359323] = -this_block_jacobian[359324];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[359950] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[359951] = -this_block_jacobian[359950]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[360576] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[360575] = -this_block_jacobian[360576]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[361202] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[361203] = this_block_jacobian[361202]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[361829] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[361827] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[361828] = -this_block_jacobian[361827] - 
        this_block_jacobian[361829]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[362454] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[362453] = -this_block_jacobian[362454];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[363080] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[363081] = -this_block_jacobian[363080]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[363706] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[363705] = -this_block_jacobian[363706]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[364332] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[364333] = this_block_jacobian[364332]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[364959] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[364957] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[364958] = -this_block_jacobian[364957] - 
        this_block_jacobian[364959]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[365584] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[365583] = -this_block_jacobian[365584];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[366210] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[366211] = -this_block_jacobian[366210]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[366836] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[366835] = -this_block_jacobian[366836]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[367462] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[367463] = this_block_jacobian[367462]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[368089] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[368087] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[368088] = -this_block_jacobian[368087] - 
        this_block_jacobian[368089]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[368714] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[368713] = -this_block_jacobian[368714];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[369340] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[369341] = -this_block_jacobian[369340]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[369966] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[369965] = -this_block_jacobian[369966]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[370592] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[370593] = this_block_jacobian[370592]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[371219] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[371217] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[371218] = -this_block_jacobian[371217] - 
        this_block_jacobian[371219]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[371844] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[371843] = -this_block_jacobian[371844];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[372470] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[372471] = -this_block_jacobian[372470]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[373096] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[373095] = -this_block_jacobian[373096]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[373722] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[373723] = this_block_jacobian[373722]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[374349] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[374347] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[374348] = -this_block_jacobian[374347] - 
        this_block_jacobian[374349]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[374974] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[374973] = -this_block_jacobian[374974];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[375600] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[375601] = -this_block_jacobian[375600]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[376226] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[376225] = -this_block_jacobian[376226]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[376852] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[376853] = this_block_jacobian[376852]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[377479] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[377477] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[377478] = -this_block_jacobian[377477] - 
        this_block_jacobian[377479]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[378104] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[378103] = -this_block_jacobian[378104];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[378730] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[378731] = -this_block_jacobian[378730]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[379356] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[379355] = -this_block_jacobian[379356]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[379982] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[379983] = this_block_jacobian[379982]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[380609] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[380607] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[380608] = -this_block_jacobian[380607] - 
        this_block_jacobian[380609]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[381234] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[381233] = -this_block_jacobian[381234];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[381860] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[381861] = -this_block_jacobian[381860]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[382486] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[382485] = -this_block_jacobian[382486]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[383112] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[383113] = this_block_jacobian[383112]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[383739] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[383737] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[383738] = -this_block_jacobian[383737] - 
        this_block_jacobian[383739]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[384364] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[384363] = -this_block_jacobian[384364];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[384990] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[384991] = -this_block_jacobian[384990]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[385616] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[385615] = -this_block_jacobian[385616]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[386242] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[386243] = this_block_jacobian[386242]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[386869] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[386867] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[386868] = -this_block_jacobian[386867] - 
        this_block_jacobian[386869]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[387494] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[387493] = -this_block_jacobian[387494];//He+ : 9-alpha_(He++)ne
       
    // H0
    this_block_jacobian[388120] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[388121] = -this_block_jacobian[388120]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    
    //H+
    this_block_jacobian[388746] = -constants[2]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[388745] = -this_block_jacobian[388746]; // H0 : 2-alpha_(H+)ne
    
    // He0
    this_block_jacobian[389372] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[389373] = this_block_jacobian[389372]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    
    // He+
    this_block_jacobian[389999] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[389997] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[389998] = -this_block_jacobian[389997] - 
        this_block_jacobian[389999]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    
    // He++
    this_block_jacobian[390624] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[390623] = -this_block_jacobian[390624];//He+ : 9-alpha_(He++)ne
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
