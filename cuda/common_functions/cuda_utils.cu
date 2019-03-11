#include <stdio.h>
#include <cublas_v2.h>
#include "cuda_utils.h"

__device__ int get_system_bid(){
    return blockIdx.z*gridDim.y + blockIdx.y;
}

__global__ void cudaRoutineFlat(int Neqn_p_sys, float * d_arr){
    printf("float: %d - %.3f\n",threadIdx.x,d_arr[threadIdx.x]);
}

__global__ void cudaRoutineFlatInt(int Neqn_p_sys, int * d_arr){
    printf("int: %d - %d\n",threadIdx.x,d_arr[threadIdx.x]);
}
__global__ void cudaRoutine(int Neqn_p_sys, float ** d_arr,int index){
    printf("%d - %.2f hello\n",threadIdx.x,d_arr[index][threadIdx.x]);
}

__global__ void printfCUDA(float * pointer){
    printf("%f value of cuda pointer \n",*pointer);
}

__global__ void printFloatArrayCUDA(float * pointer, int Narr){
    // safety in case it's called with a bunch of threads lol
    if (threadIdx.x == 0 && blockIdx.x == 0){
        for (int i = 0; i<Narr; i++){
            printf("%.2e \t",pointer[i]);
        }
        printf("\n");
    }
}

__global__ void checkCublasINFO(
    int * INFO,
    int * bool_flag,
    int Nsystems){
    // replace the values of v1 with the error
    int bid = get_system_bid();

    if (bid < Nsystems){
        if (INFO[bid]){
            *bool_flag = 1;
        }
    }
}

void checkCublasErrorState(int * INFO,int * d_INFO_bool,int INFO_bool,int Nsystems, dim3 ode_gridDim){
    checkCublasINFO<<<ode_gridDim,1>>>(INFO, d_INFO_bool,Nsystems);
    cudaMemcpy(&INFO_bool,&d_INFO_bool,sizeof(int),cudaMemcpyDeviceToHost);
    printf("INFO: %d \n",INFO_bool);
    INFO_bool = 0;
    cudaMemcpy(d_INFO_bool,&INFO_bool,sizeof(int),cudaMemcpyHostToDevice);
}

const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

