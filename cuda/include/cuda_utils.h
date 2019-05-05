#include <cublas_v2.h>

__global__ void cudaRoutineFlat(int, float *);

__global__ void cudaRoutineFlatInt(int,int *);

__global__ void cudaRoutine(int, float **,int);

__global__ void printfCUDA(float *);

__global__ void printFloatArrayCUDA(float *, int);

__global__ void checkCublasINFO(int *, int *, int);

__global__ void gjeInvertMatrixBatched(
    float *, // d_matricess_flat,
    int ,//Ndim,
    int );//Nbatch)

const char *_cudaGetErrorEnum(cublasStatus_t);

void checkCublasErrorState(int *,int *,int,int, dim3);
