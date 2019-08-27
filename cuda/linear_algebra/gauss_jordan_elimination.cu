#include <stdio.h>
#include "config.h"
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
    WindFloat * row_array, 
    int Ndim,
    int place_index, 
    WindFloat * scale_factor_out,
    bool read_scale_factor,
    WindFloat * shared_array){

    if (read_scale_factor){
        if (threadIdx.x==0){
            shared_array[0] = row_array[place_index];
        }
        __syncthreads();
        *scale_factor_out = shared_array[0];
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
    WindFloat * target_row_array,
    WindFloat * row_array, 
    int Ndim,
    int place_index, 
    WindFloat * scale_factor_out,
    bool read_scale_factor,
    WindFloat * shared_array){

    // were we passed a scale_factor to scale by 
    //  or should we read it from the column we're 
    //  zeroing? 

    if (read_scale_factor){
        if (threadIdx.x==0){
            shared_array[0] = target_row_array[place_index];
        }
        __syncthreads();
        *scale_factor_out = shared_array[0];
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
    WindFloat * d_this_matrix_flat,
    WindFloat * d_inverse_matrix_flat,
    int Ndim,
    WindFloat * shared_array){

    // allocate a place to store row scale factors
    //  so that they may be applied to the d_inverse_matrix
    WindFloat d_this_row_scale_factor;

    // put this matrix into upper triangular form
    for (int row_i=0; row_i<Ndim;row_i++){
        scaleRow(
            d_this_matrix_flat + row_i*Ndim, // this row
            Ndim, // how many elements in row
            row_i, // which column am i dividing by
            &d_this_row_scale_factor,
            true,
            shared_array); 

        // apply the same transformation to the inverse
        scaleRow(
            d_inverse_matrix_flat + row_i*Ndim, // this row
            Ndim,
            NULL,// use provided scale factor
            &d_this_row_scale_factor,
            false,
            shared_array);

        for (int next_row_i=row_i+1; next_row_i < Ndim; next_row_i++){
            subtractRows(
                d_this_matrix_flat + next_row_i*Ndim,
                d_this_matrix_flat + row_i*Ndim,
                Ndim,
                row_i, // which column am I zeroing out
                &d_this_row_scale_factor,
                true,
                shared_array);

            subtractRows(
                d_inverse_matrix_flat + next_row_i*Ndim,
                d_inverse_matrix_flat + row_i*Ndim,
                Ndim,
                NULL, // use provided scale factor
                &d_this_row_scale_factor,
                false,
                shared_array);
        }
    }
}

__device__ void gjeLFactor(
    WindFloat * d_this_matrix_flat,
    WindFloat * d_inverse_matrix_flat,
    int Ndim,
    WindFloat * shared_array){

    //  so that they may be applied to the d_inverse_matrix
    WindFloat d_this_row_scale_factor;

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
                &d_this_row_scale_factor,
                true,
                shared_array);

            subtractRows(
                d_inverse_matrix_flat + bnri*Ndim,
                d_inverse_matrix_flat + bri*Ndim,
                Ndim,
                NULL, // use provided scale factor
                &d_this_row_scale_factor,
                false,
                shared_array);
        }
    }
}

__device__ void setIdentity(
    WindFloat * d_inverse_matrix_flat,
    int Ndim){

    int tid;
    // loop over whole matrix to set zeros
    for (int iterations=0; iterations<(Ndim*Ndim/blockDim.x); iterations++){
        tid = getGJETID(iterations);
        if (tid < Ndim*Ndim){
            d_inverse_matrix_flat[tid]=0;    
        }
    }

    __syncthreads();
    // loop over the diagonal
    for (int iterations=0; iterations<(Ndim/blockDim.x); iterations++){
        tid = getGJETID(iterations);
        if (tid < Ndim){
            d_inverse_matrix_flat[tid*(Ndim+1)] = 1.0;
        }
    }
}

__device__ void gjeInvertMatrix(
    WindFloat * d_this_matrix_flat,
    WindFloat * d_inverse_matrix_flat,
    int Ndim,
    WindFloat * shared_array){

    // allocate a temporary inverse matrix
    //extern __shared__ WindFloat d_inverse_matrix_flat[];

    // generate an identity matrix in the shared inverse matrix 
    setIdentity(d_inverse_matrix_flat,Ndim);
    gjeUFactor(d_this_matrix_flat,d_inverse_matrix_flat,Ndim,shared_array);
    gjeLFactor(d_this_matrix_flat,d_inverse_matrix_flat,Ndim,shared_array);
        
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
    WindFloat * d_matricess_flat,
    WindFloat * d_inverse_matricess_flat,
    int Ndim,
    int Nbatch){

    int bid = getGJEBID();
    __shared__ WindFloat shared_array[1];
    gjeInvertMatrix(
        d_matricess_flat + bid*Ndim*Ndim,
        d_inverse_matricess_flat + bid*Ndim*Ndim,
        Ndim,
        shared_array);
}
