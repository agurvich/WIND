#include <stdio.h>
#include "cu_routines.h"

__global__ void hello(int * a, int * b){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    printf("tid %d\n",tid);
    a[tid]=a[tid]+b[tid];
} // hello

__device__ void calculate_dydt(
    float tnow,
    float * constants,
    float * equations,
    float * dydt){
    // assumes that constant and equation are pointers that start 
    //  at the beginning of the block's values.
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    dydt[threadIdx.x] = constants[tid];
} // calculate_dydyt

__device__ void euler_innerstep(
    float tnow, // the current time
    float tend, // the time we want to stop
    float h, // the step size to use 
    float * constants, // the constants for each system
    float * equations, // a flattened array containing the y value for each equation in each system
    int Nsystems, // the number of systems
    int Nequations_per_system,
    float * shared_dydt){ // the number of equations in each system

    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    //dydt = (float *) malloc(blockDim.x*sizeof(float));
    while (tnow < tend){
        // limit step size based on remaining time
        h = fmin(tend - tnow, h);

        //calculate the derivatives
        calculate_dydt(
            tnow,
            constants,
            equations,
            shared_dydt);

        // update value of equations
        equations[tid] += h*shared_dydt[threadIdx.x];

        tnow+=h;
    } // while(tnow < tend)

}// euler_innerstep

__device__ int check_error(float y1, float y2){
    float delta = (y2-y1);
    return (delta < ABSOLUTE_TOLERANCE && delta/y2 < RELATIVE_TOLERANCE);
}

__device__ void recursiveWrapper(
    float tnow, // the current time
    float tend, // the time we integrating the system until
    float * constants, // the constants for each system
    float * equations, // a flattened array containing the y value for each equation in each system
    int Nsystems, // the number of systems
    int Nequations_per_system, // the number of equations in each system
    int recursion_depth, // the number of times this function has been called recursively
    int max_refinement_depth){ // the maximum number of times this function can be called recursively){

    int error_flag = 0;
        
    //printf("%d thread %d block\n",threadIdx.x,blockIdx.x);
    if (tid < Nsystems*Nequations_per_system ) {
        // solve for this step
        if (recursion_depth < 1){
            euler_innerstep(
                tnow, tend,
                tnow-tend,
                constants, equations,
                y1,
                Nsystems, Nequations_per_system,
                shared_dydt);
        }
        else{
            // have the first thread in the block swap the pointers
            if (threadIdx.x == 0){
                float * temporary_hold;
                temporary_hold = &y2;
                y2 = &y1;
                y1 = &temporary_hold;
            }
        }
        __syncthreads();
        // solve the same step but at half the stepsize
        euler_innerstep(
            tnow, tend,
            (tnod-tend)/2.0,
            constants, equations,
            y2,
            Nsystems, Nequations_per_system,
            shared_dydt);
        __syncthreads();

        // compare the two and determine if we need to refine
        error_flag = check_error(y1[threadIdx.x],y2[threadIdx.x]);
        __syncthreads();

     }
    if (error_flag > 0 && recursion_depth <= max_refinement_depth){
        return recursiveWrapper(recursion_depth +1); // TODO need to pass y2 as the new y1 and not calculate it
    }
    else{
        // use local extrapolation, the solution to h^3 error is 
        // y2 + delta, see NR 17.2.23 applied to euler equation
        // TODO make sure that this is correctly unpacking this system's 
        // values into the correct global memory
        return 2*y2[threadIdx.x] - y1[threadIdx.x];
    }


}

__global__ void integrate_euler(
    float tnow, // the current time
    float tend, // the time we integrating the system until
    float * constants, // the constants for each system
    float * equations, // a flattened array containing the y value for each equation in each system
    int Nsystems, // the number of systems
    int Nequations_per_system, // the number of equations in each system
    int max_refinement_depth // the maximum number of times this function can be called recursively){
    ){

    // unique thread ID , based on local ID in block and block ID
    int tid = threadIdx.x + ( blockDim.x * blockIdx.x);

    // ensure thread within limit

    extern __shared__ float total_shared[];
    // total_shared is a pointer to the beginning of this block's shared
    //  memory. If we want to use multiple shared memory arrays we must
    //  manually offset them within that block and allocate enough memory
    //  when initializing the kernel (<<dimGrid,dimBlock,sbytes>>)
    float * shared_dydt = (float *) &total_shared[0];

    // make two arrays to hold the solution for this system
    // TODO should probably be shared memory
    float * y0 = (float *) malloc(sizeof(float)*Nequations_per_system);
    float * y1 = (float *) malloc(sizeof(float)*Nequations_per_system);
    float * y2 = (float *) malloc(sizeof(float)*Nequations_per_system);

    recursiveWrapper(
        tnow,tend,
        constants,equations,
        Nsystems,Nequations_per_system
        0,max_refinement_depth,
        y0,y1,y2);

} //integrate_euler
