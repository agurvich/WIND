#include "vector_kernels.h"

__global__ void overwriteVector(float * v1, float * v2){
    // copies the contents of v1 into v2
    v2[blockIdx.x*blockDim.x+threadIdx.x] = v1[blockIdx.x*blockDim.x+threadIdx.x];
}

__global__ void scaleVector(float * vector, float * scales){
    // assumes that gridDim = Nsystems and blockDim = Neqn_p_sys
    vector[blockIdx.x*blockDim.x+threadIdx.x]*=scales[blockIdx.x];
}
__global__ void addVectors(float alpha, float * v1, float beta, float * v2, float * v3){
    // outputs the result in v3
    v3[blockIdx.x*blockDim.x+threadIdx.x] = alpha * v1[blockIdx.x*blockDim.x+threadIdx.x] + beta * v2[blockIdx.x*blockDim.x+threadIdx.x];
}

__global__ void addArrayToBatchArrays(float ** single_arr, float ** batch_arrs, float alpha, float beta, float *p_beta){
    // assumes that gridDim = Nsystems and blockDim = Neqn_p_sys
    batch_arrs[blockIdx.x][threadIdx.x]=alpha*single_arr[0][threadIdx.x]+ beta*(*p_beta)*batch_arrs[blockIdx.x][threadIdx.x];
}

__global__ void updateTimestep(float * timestep, float * derivatives_flat, float * scale_factor, int * max_index){
    // changes the value of the pointer in global memory on the device without copying back the derivatives
    //float ABSOLUTE_TOLERANCE = 1e-4;
    // -1 because cublas is 1 index. whyyyy
    /*
    if ( derivatives_flat[*max_index-1] < 1000*ABSOLUTE_TOLERANCE){
        *timestep = 1000*ABSOLUTE_TOLERANCE;
    }
    else{
        *timestep = ABSOLUTE_TOLERANCE/derivatives_flat[*max_index-1];
    }
    */
    *timestep = .01 * (*scale_factor);

    // TODO fixed timestep because why not?
    //*timestep = 0.25;
}
