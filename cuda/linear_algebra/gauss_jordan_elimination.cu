#include <stdio.h>
#include "linear_algebra.h"

__device__ int getGJETID(int iterations){
    //return blockIdx.x*blockDim.x+threadIdx.x;
    return iterations*blockDim.x + threadIdx.x;
}

__device__ int getGJEBID(){
    // assumes that gridDim.y = Nsystems, and blockDim.x = Neqn_p_sys
    //return blockIdx.z*gridDim.y + blockIdx.y;
    return blockIdx.x;//*blockDim.x;
}

__device__ void scaleRow(
    float * row_array, 
    int Ndim,
    int place_index, 
    float * scale_factor_out,
    bool read_scale_factor){

    if (read_scale_factor){
        *scale_factor_out = row_array[place_index];
    }

    int tid;
    for (int iterations=0; iterations<(Ndim/blockDim.x); iterations++){
        tid = getGJETID(iterations);
        if (tid < Ndim){
            row_array[tid]/=(*scale_factor_out);
        }
    }
    __syncthreads();

}

__device__ void subtractRows(
    float * target_row_array,
    float * row_array, 
    int Ndim,
    int place_index, 
    float * scale_factor_out,
    bool read_scale_factor){

    // were we passed a scale_factor to scale by 
    //  or should we read it from the column we're 
    //  zeroing? 
    if (read_scale_factor){
        *scale_factor_out = target_row_array[place_index];
    }

    int tid;
    // loop over a row
    for (int iterations=0; iterations<(Ndim/blockDim.x); iterations++){
        tid = getGJETID(iterations);
        if (tid < Ndim){
            target_row_array[tid] -= (*scale_factor_out)*row_array[tid];
        }
    }

    // make sure everyone finishes before moving on 
    __syncthreads();
}

__device__ void gjeUFactor(
    float * d_this_matrix_flat,
    float * d_inverse_matrix_flat,
    int Ndim){

    // allocate a place to store row scale factors
    //  so that they may be applied to the d_inverse_matrix
    float d_this_row_scale_factor[1];

    // put this matrix into upper triangular form
    for (int row_i=0; row_i<Ndim;row_i++){
        scaleRow(
            d_this_matrix_flat + row_i*Ndim, // this row
            Ndim, // how many elements in row
            row_i, // which column am i dividing by
            d_this_row_scale_factor,
            true); 

        // apply the same transformation to the inverse
        scaleRow(
            d_inverse_matrix_flat + row_i*Ndim, // this row
            Ndim,
            NULL,// use provided scale factor
            d_this_row_scale_factor,
            false);

        for (int next_row_i=row_i+1; next_row_i < Ndim; next_row_i++){
            subtractRows(
                d_this_matrix_flat + next_row_i*Ndim,
                d_this_matrix_flat + row_i*Ndim,
                Ndim,
                row_i, // which column am I zeroing out
                d_this_row_scale_factor,
                true);

            subtractRows(
                d_inverse_matrix_flat + next_row_i*Ndim,
                d_inverse_matrix_flat + row_i*Ndim,
                Ndim,
                NULL, // use provided scale factor
                d_this_row_scale_factor,
                false);
        }
    }
}

__device__ void gjeLFactor(
    float * d_this_matrix_flat,
    float * d_inverse_matrix_flat,
    int Ndim){

    //  so that they may be applied to the d_inverse_matrix
    float d_this_row_scale_factor[1];

    int bri;
    int bnri;
    // put it into lower triangular form, start from the bottom
    for (int row_i=0; row_i<Ndim;row_i++){
        bri = Ndim - 1 - row_i;
        for (int next_row_i=row_i+1; next_row_i < Ndim; next_row_i++){
            bnri = Ndim - 1 - next_row_i;
            subtractRows(
                d_this_matrix_flat + bnri*Ndim,
                d_this_matrix_flat + bri*Ndim,
                Ndim,
                bri, // which column am I zeroing out
                d_this_row_scale_factor,
                true);

            subtractRows(
                d_inverse_matrix_flat + bnri*Ndim,
                d_inverse_matrix_flat + bri*Ndim,
                Ndim,
                NULL, // use provided scale factor
                d_this_row_scale_factor,
                false);
        }
    }
}

__device__ void setIdentity(
    float * d_inverse_matrix_flat,
    int Ndim){

    int tid;
    // loop over whole matrix to set zeros
    for (int iterations=0; iterations<(Ndim*Ndim/blockDim.x); iterations++){
        tid = getGJETID(iterations);
        if (tid < Ndim*Ndim){
            d_inverse_matrix_flat[tid]=0;    
        }
    }

    // loop over the diagonal
    for (int iterations=0; iterations<(Ndim/blockDim.x); iterations++){
        tid = getGJETID(iterations);
        if (tid < Ndim){
            d_inverse_matrix_flat[tid*(Ndim+1)] = 1.0;
        }
    }
}

__device__ void gjeInvertMatrix(
    float * d_this_matrix_flat,
    float * d_inverse_matrix_flat,
    int Ndim){

    // allocate a temporary inverse matrix
    //extern __shared__ float d_inverse_matrix_flat[];

    // generate an identity matrix in the shared inverse matrix 
    setIdentity(d_inverse_matrix_flat,Ndim);

    gjeUFactor(d_this_matrix_flat,d_inverse_matrix_flat,Ndim);

    gjeLFactor(d_this_matrix_flat,d_inverse_matrix_flat,Ndim);
        
    /*
    // copy the output back
    int tid;
    for (int iterations=0; iterations<(Ndim*Ndim/blockDim.x); iterations++){
        tid = getGJETID(iterations);
        if (tid < Ndim*Ndim){
            d_this_matrix_flat[tid] = d_inverse_matrix_flat[tid];
        }
    }
    */
}

__global__ void gjeInvertMatrixBatched(
    float * d_matricess_flat,
    float * d_inverse_matricess_flat,
    int Ndim,
    int Nbatch){

    int bid = getGJEBID();
    gjeInvertMatrix(
        d_matricess_flat + bid*Ndim*Ndim,
        d_inverse_matricess_flat + bid*Ndim*Ndim,
        Ndim);
}
