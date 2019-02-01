// SIE_solver
__global__ void cudaRoutineFlat(int , float * );
__global__ void cudaRoutine(int , float ** ,int );
__global__ void printfCUDA(float * );
void setIdentityMatrix(float * ,int );
float ** initializeDeviceMatrix(float * , float ** , int ,int );
__global__ void addArrayToBatchArrays(float ** , float ** , float , float, float *);
__global__ void scaleVector(float * , float * );
__global__ void updateTimestep(float * , float * , int * );
__global__ void calculateDerivatives(float * , float );
__global__ void calculateJacobians(float **, float );

// BDF2_solver
