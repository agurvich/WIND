#include "ode.h"

__global__ void calculateDerivatives(float * d_derivatives_flat, float time){
    d_derivatives_flat[threadIdx.x + blockIdx.x*blockDim.x] = (threadIdx.x + 1) * time * time;
}

__global__ void calculateJacobians(float **d_Jacobianss, float time){
    // relies on the fact that blockDim.x is ndim and we're striding to get to the diagonals
    d_Jacobianss[blockIdx.x][threadIdx.x+blockDim.x*threadIdx.x] = ((float ) (threadIdx.x+1)) * time;
    //printf("%d - %d = %d\n",blockIdx.x,threadIdx.x+blockDim.x*threadIdx.x,threadIdx.x+1);
}
