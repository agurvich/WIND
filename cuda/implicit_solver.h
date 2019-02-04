// SIE_solver
__global__ void cudaRoutineFlat(int , float * );
__global__ void cudaRoutine(int , float ** ,int );
__global__ void printfCUDA(float * );
void setIdentityMatrix(float * ,int );
float ** initializeDeviceMatrix(float * , float ** , int ,int );
__global__ void addArrayToBatchArrays(float ** , float ** , float , float, float *);
__global__ void addVectors(float , float * , float , float *, float *);
__global__ void overwriteVector(float *, float *);
__global__ void scaleVector(float * , float * );
__global__ void updateTimestep(float * , float * , float *, int * );
__global__ void calculateDerivatives(float * , float );
__global__ void calculateJacobians(float **, float );

void SIE_step(
    float * , // pointer to current time
    float * , // device pointer to the current timestep (across all systems, lame!!)
    float ** ,  // Nsystems x Neqn_p_sys*Neqn_p_sys 2d array with flattened jacobians
    float ** , // Nsystems x Neqn_p_sys*Neqn_p_sys 2d array to store output (same as jacobians to overwrite)
    float ** , // 1 x Neqn_p_sys*Neqn_p_sys array storing the identity (ideally in constant memory?)
    float ** , // Nsystems x Neqn_p_sys 2d array to store derivatives
    float * , // Nsystems*Neqn_p_sys 1d array (flattened above)
    float * , // output state vector, iterative calls integrates
    int , // number of ODE systems
    int ); // number of equations in each system


// BDF2_solver
