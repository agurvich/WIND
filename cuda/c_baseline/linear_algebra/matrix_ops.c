#include <string.h>
#include <stdlib.h>

#include "linear_algebra.h"

void matrixVectorMult(
    float * matrix,
    float * vector,
    int ndim){
    
    float * out; 
    out = malloc(sizeof(float)*ndim);
    memset(out,0,sizeof(float)*ndim);

    // assumes that the matrix is transposed
    for (int col_i=0; col_i<ndim; col_i++){
        for (int row_i=0; row_i<ndim; row_i++){
            out[row_i]+=matrix[col_i*ndim + row_i]*vector[col_i];
        }
    }

    // copy the output back to the vector
    for (int row_i=0; row_i<ndim; row_i++){
        vector[row_i] = out[row_i];
    }
    free(out);
}
