#include <stdio.h>

__device__ int getTID(int iterations){
    //return blockIdx.x*blockDim.x+threadIdx.x;
    return iterations*blockDim.x + threadIdx.x;
}

__device__ int getBID(){
    // assumes that gridDim.y = Nsystems, and blockDim.x = Neqn_p_sys
    //return blockIdx.z*gridDim.y + blockIdx.y;
    return blockIdx.x;//*blockDim.x;
}

__device__ void scaleRow(
    float * row_array, 
    int Ndim,
    int place_index, 
    float * scale_factor_out){

    if (place_index != NULL){
        *scale_factor_out = row_array[place_index];
    }

    int tid;
    for (int iterations=0; iterations<(Ndim/blockDim.x); iterations++){
        tid = getTID(iterations);
        if (tid < Ndim){
            row_array[tid]/=(*scale_factor_out);
        }
    }
}


__device__ void subtractRows(
    float * target_row_array,
    float * row_array, 
    int Ndim,
    int place_index, 
    float * scale_factor_out){

    // were we passed a scale_factor to scale by 
    //  or should we read it from the column we're 
    //  zeroing? 
    if (place_index != NULL){
        *scale_factor_out = target_row_array[place_index];
    }

    int tid;
    for (int iterations=0; iterations<(Ndim/blockDim.x); iterations++){
        tid = getTID(iterations);
        if (tid < Ndim){
            target_row_array[tid] -= (*scale_factor_out)*row_array[tid];
        }
    }

    // make sure everyone finishes before moving on 
    __syncthreads();
}

__device__ void gjeInvertMatrix(
    float * d_this_matrix_flat,
    float * d_identity_flat,
    int Ndim,
    int system_start_index){

    // allocate a temporary inverse matrix
    __shared__ float d_inverse_matrix_flat;
    //TODO size of this matrix at runtime ? 

    // copy the identity matrix to the shared inverse matrix 
    int tid;
    for (int iterations=0; iterations<(Ndim*Ndim/blockDim.x); iterations++){
        tid = getTID(iterations);
        if (tid < Ndim){
            d_inverse_matrix_flat[tid] = d_identity_flat[tid];
        }
    }

    // allocate a place to store row scale factors
    //  so that they may be applied to the d_inverse_matrix
    float * d_this_row_scale_factor;

    // put this matrix into upper triangular form
    for (int row_i=0; row_i<Ndim;row_i++){
        scaleRow(
            d_this_matrix_flat + row_i*Ndim, // this row
            Ndim, // how many elements in row
            row_i,
            d_this_row_scale_factor); // outputs row[row_i] to d_this_row_scale_factor

        // apply the same transformation to the inverse
        scaleRow(
            d_inverse_matrix_flat + row_i*Ndim, // this row
            Ndim,
            NULL,
            d_this_row_scale_factor);

        for (int next_row_i=0; next_row_i < Ndim; next_row_i++){
            subtractRows(
                d_this_matrix_flat + next_row_i*Ndim,
                d_this_matrix_flat + row_i*Ndim,
                Ndim,
                row_i, // which column am I zeroing out
                d_this_row_scale_factor);

            subtractRows(
                d_inverse_matrix_flat + next_row_i*Ndim,
                d_inverse_matrix_flat + row_i*Ndim,
                Ndim,
                NULL, // use provided scale factor
                d_this_row_scale_factor);
        }
    }

    int bri;
    int bnri;
    // put it into lower triangular form, start from the bottom
    for (int row_i=0; row_i<Ndim;row_i++){
        bri = Ndim - 1 - row_i;
        for (int next_row_i=0; next_row_i < Ndim; next_row_i++){
            bnri = N - 1 - next_row_i;
            subtractRows(
                d_this_matrix_flat + bnri*Ndim,
                d_this_matrix_flat + bri*Ndim,
                Ndim,
                row_i, // which column am I zeroing out
                d_this_row_scale_factor);

            subtractRows(
                d_inverse_matrix_flat + bnri*Ndim,
                d_inverse_matrix_flat + bri*Ndim,
                Ndim,
                NULL, // use provided scale factor
                d_this_row_scale_factor);
        }
        
    }

    // copy the output back
    int tid;
    for (int iterations=0; iterations<(Ndim*Ndim/blockDim.x); iterations++){
        tid = getTID(iterations);
        if (tid < Ndim){
            d_this_matrix_flat[tid] = d_inverse_matrix_flat[tid];
        }
    }
}

__global__ void gjeInvertMatrixBatched(
    float * d_matricess_flat,
    float * d_identity_flat,
    int Ndim,
    int Nbatch){

    int bid = getBID();
    gjeInvertMatrix(
        d_matricess_flat + bid*Ndim*Ndim,
        d_identity_flat,
        Ndim);
}
