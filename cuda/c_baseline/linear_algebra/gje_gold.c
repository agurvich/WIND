#include <stdlib.h>
#include <string.h>
#include "linear_algebra.h"

void setIdentity(
    float * inverse,
    int ndim){

    // zero out the inverse to the identity for future gje
    memset(inverse,0,sizeof(float)*ndim*ndim);
    int diag_index;
    for (int eqn_i=0; eqn_i<ndim; eqn_i++){
        diag_index = eqn_i*(ndim+1);
        inverse[diag_index] = 1;
    }
}

void scaleRow(
    float * row_array, 
    int ndim,
    int place_index, 
    float * scale_factor_out,
    int read_scale_factor){

    if (read_scale_factor){
        *scale_factor_out = row_array[place_index];
    }

    for (int tid=0; tid<(ndim); tid++){
        row_array[tid]/=(*scale_factor_out);
    }
}

void subtractRows(
    float * target_row_array,
    float * row_array, 
    int ndim,
    int place_index, 
    float * scale_factor_out,
    int read_scale_factor){

    // were we passed a scale_factor to scale by 
    //  or should we read it from the column we're 
    //  zeroing? 

    if (read_scale_factor){
        *scale_factor_out = target_row_array[place_index];
    }

    // loop over a row
    for (int tid=0; tid<ndim; tid++){
        target_row_array[tid] -= (*scale_factor_out)*row_array[tid];
    }
}

void gjeUFactor(
    float * matrix,
    float * inverse,
    int ndim){

    // allocate a place to store row scale factors
    //  so that they may be applied to the d_inverse_matrix
    float this_row_scale_factor;

    // put this matrix into upper triangular form
    for (int row_i=0; row_i<ndim;row_i++){
        scaleRow(
            matrix + row_i*ndim, // this row
            ndim, // how many elements in row
            row_i, // which column am i dividing by
            &this_row_scale_factor,
            1); 

        // apply the same transformation to the inverse
        scaleRow(
            inverse + row_i*ndim, // this row
            ndim,
            -1,// use provided scale factor
            &this_row_scale_factor,
            0);

        for (int next_row_i=row_i+1; next_row_i < ndim; next_row_i++){
            subtractRows(
                matrix + next_row_i*ndim,
                matrix + row_i*ndim,
                ndim,
                row_i, // which column am I zeroing out
                &this_row_scale_factor,
                1);

            subtractRows(
                inverse + next_row_i*ndim, // float * target_row_array,
                inverse + row_i*ndim, // float * row_array, 
                ndim, // int ndim,
                -1, // use provided scale factor int place_index, 
                &this_row_scale_factor, // float * scale_factor_out,
                0); // int read_scale_factor){
        }
    }
}

void gjeLFactor(
    float * matrix,
    float * inverse,
    int ndim){

    //  so that they may be applied to the d_inverse_matrix
    float this_row_scale_factor;

    int bri;
    int bnri;
    // put it into lower triangular form, start from the bottom
    for (int row_i=0; row_i<ndim;row_i++){
        bri = ndim - 1 - row_i;
        for (int next_row_i=row_i+1; next_row_i < ndim; next_row_i++){
            bnri = ndim - 1 - next_row_i;
            subtractRows(
                matrix + bnri*ndim,
                matrix + bri*ndim,
                ndim,
                bri, // which column am I zeroing out
                &this_row_scale_factor,
                1);

            subtractRows(
                inverse + bnri*ndim,
                inverse + bri*ndim,
                ndim,
                -1, // use provided scale factor
                &this_row_scale_factor,
                0);
        }
    }
}

void gjeInvertMatrixGold(
    float * matrix,
    float * inverse,
    int ndim){

    setIdentity(inverse,ndim);
    gjeUFactor(matrix,inverse,ndim);
    gjeLFactor(matrix,inverse,ndim);
}
