
__global__ void calculateDerivatives(float * , float *, float *, int, int, float);

__global__ void calculateJacobians(float **, float *, float *, int, int, float);

__global__ void read_texture(void *);

void resetSystem(float**, float *, float **, float *, 
    float *,float *,float *, int, int, float);

void configureGrid(
    int, int, 
    int * , 
    dim3 *, 
    dim3 *, 
    dim3 *);

extern void * RHS_input;
#define NUM_CONST 2
