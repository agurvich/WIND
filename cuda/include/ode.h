
__global__ void calculateDerivatives(float * , float *, float *, int, int, float);

__global__ void calculateJacobians(float **, float *, float *, int, int, float);

void resetSystem(float**, float *, float **, float *, 
    float *,float *,float *, int, int, float);

void configureGrid(
    int, int, 
    int * , 
    dim3 *, 
    dim3 *, 
    dim3 *);
#define NUM_CONST 10
