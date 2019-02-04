#include <stdio.h>
#include "cuda_utils.h"

__global__ void cudaRoutineFlat(int Neqn_p_sys, float * d_arr){
    printf("%d - %.3f hello\n",threadIdx.x,d_arr[threadIdx.x]);
}
__global__ void cudaRoutine(int Neqn_p_sys, float ** d_arr,int index){
    printf("%d - %.2f hello\n",threadIdx.x,d_arr[index][threadIdx.x]);
}

__global__ void printfCUDA(float * pointer){
    printf("%f value of cuda pointer \n",*pointer);
}
