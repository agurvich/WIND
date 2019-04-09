
// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>
#include "explicit_solver.h"
#include "ode.h"
  
void printArray(int * arr,int N){
    for (int i = 0; i<N;i++){
        printf("%d ",arr[i]);
    }
    printf("\n");
}

int cudaIntegrateRK2(
    float tnow, // the current time
    float tend, // the time we integrating the system to
    float * constants, // the constants for each system
    float * equations, // a flattened array containing the y value for each equation in each system
    int Nsystems, // the number of systems
    int Nequations_per_system){ // the number of equations in each system

#ifdef LOUD
    printf("RK2 Received %d systems, %d equations per system\n",Nsystems,Nequations_per_system);
#endif

    // copy the arrays over to the device
    int Nequations = Nsystems*Nequations_per_system;
    int equations_size = Nequations*sizeof(float);

    float *constantsDevice;
    cudaMalloc((void**)&constantsDevice, equations_size); 
    cudaMemcpy( constantsDevice, constants, NUM_CONST*sizeof(float), cudaMemcpyHostToDevice ); 

    float *equationsDevice;
    cudaMalloc((void**)&equationsDevice, equations_size); 
    cudaMemcpy( equationsDevice, equations, equations_size, cudaMemcpyHostToDevice ); 

    int nloops=0;
    int * nloopsDevice;
    cudaMalloc(&nloopsDevice, sizeof(int)); 
    cudaMemcpy(nloopsDevice, &nloops, sizeof(int), cudaMemcpyHostToDevice ); 


    float * tnowDevice;
    cudaMalloc(&tnowDevice, sizeof(float)); 
    cudaMemcpy(tnowDevice, &tnow, sizeof(float), cudaMemcpyHostToDevice ); 

    float * tendDevice;
    cudaMalloc(&tendDevice, sizeof(float)); 
    cudaMemcpy(tendDevice, &tend, sizeof(float), cudaMemcpyHostToDevice ); 

    // setup the grid dimensions
    int blocksize,gridsize;
    if (Nequations_per_system < THREAD_BLOCK_LIMIT){
        blocksize = Nequations_per_system;
        gridsize = Nsystems;
    }
    else{
        blocksize = THREAD_BLOCK_LIMIT;
        gridsize = Nequations/THREAD_BLOCK_LIMIT+1;
    }

    //printf("%d blocksize, %d gridsize\n",blocksize,gridsize);
    dim3 dimBlock( blocksize, 1 );
    dim3 dimGrid( gridsize, 1 );

    //shared mem -> 2 float arrays for each system and 1 shared flag
    integrate_rk2<<<dimGrid,dimBlock,
        Nequations_per_system*(2*sizeof(float))+ sizeof(int)
        >>> (
        tnow, tend,
        constantsDevice,equationsDevice,
        Nsystems,Nequations_per_system,
        nloopsDevice);
    
    // copy the new state back
    cudaMemcpy(equations, equationsDevice, equations_size, cudaMemcpyDeviceToHost ); 
    cudaMemcpy(&nloops,nloopsDevice,sizeof(int),cudaMemcpyDeviceToHost);
    //printf("c-equations after %.2f \n",equations[0]);

    // free up the memory on the device
    cudaFree(constantsDevice);
    cudaFree(equationsDevice);

    // return how many steps were taken
    return nloops;
} // cudaIntegrateRK2

