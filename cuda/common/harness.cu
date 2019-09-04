#include <stdio.h>
#include "config.h"
#include "ode.h"


void * d_p_RHS_input; 

int cudaIntegrateSystem(
    float tnow, // the current time
    float tend, // the time we integrating the system to
    int n_integration_steps,
    WindFloat * constants, // the constants for each system
    WindFloat * equations, // a flattened array containing the y value for each equation in each system
    int Nsystems, // the number of systems
    int Nequations_per_system, // the number of equations in each system
    float ABSOLUTE, // the absolute tolerance
    float RELATIVE){ // the relative tolerance

#ifdef LOUD
#ifdef SIE
    printf("SIE Received %d systems, %d equations per system\n",Nsystems,Nequations_per_system);
#else
    printf("RK2 Received %d systems, %d equations per system\n",Nsystems,Nequations_per_system);
#endif
#endif

    // copy the arrays over to the device
    int Nequations = Nsystems*Nequations_per_system;
    long equations_size = Nequations*sizeof(WindFloat);

    WindFloat *constantsDevice;
    cudaMalloc((void**)&constantsDevice, Nsystems*NUM_CONST*sizeof(WindFloat)); 
    cudaMemcpy( constantsDevice, constants, Nsystems*NUM_CONST*sizeof(WindFloat), cudaMemcpyHostToDevice ); 

    WindFloat *equationsDevice;
    cudaMalloc((void**)&equationsDevice, equations_size); 
    cudaMemcpy( equationsDevice, equations, equations_size, cudaMemcpyHostToDevice ); 

#ifdef SIE
    WindFloat *JacobiansDevice;
    cudaMalloc((void**)&JacobiansDevice, Nequations_per_system*equations_size); 
    //cudaMemcpy( JacobiansDevice, Jacobians, equations_size, cudaMemcpyHostToDevice ); 

    WindFloat *inversesDevice;
    cudaMalloc((void**)&inversesDevice, Nequations_per_system*equations_size); 
    //cudaMemcpy( JacobiansDevice, Jacobians, equations_size, cudaMemcpyHostToDevice ); 
#endif

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

    //Maximum number of threads per multiprocessor:  2048
    //Maximum number of threads per block:           1024
    //Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
    //Max dimension size of a grid size    (x,y,z): (2.147.483.647, 65.535, 65.535)
    int blocksize,gridsize;
    if (Nequations_per_system < THREAD_BLOCK_LIMIT){
        blocksize = Nequations_per_system;
        gridsize = Nsystems;
    }
    else{
        printf("Too many equations/system, keep it below 1024\n");
        blocksize = 0;
        gridsize = 0;
    }

    //printf("%d blocksize, %d gridsize\n",blocksize,gridsize);
    dim3 dimBlock( blocksize, 1 );
    dim3 dimGrid( gridsize, 1 );

    //shared mem -> 2 float arrays for each system and 1 shared flag
    integrateSystem<<<dimGrid,dimBlock,
        Nequations_per_system*(2*sizeof(WindFloat))+ sizeof(int)
        >>> (
        tnow, tend,
        (tend-tnow)/n_integration_steps,
        d_p_RHS_input,
        constantsDevice,equationsDevice,
#ifdef SIE
        JacobiansDevice,inversesDevice,
#else
        NULL,NULL,
#endif
        Nsystems,Nequations_per_system,
        nloopsDevice,
        ABSOLUTE,RELATIVE);
    
    // copy the new state back
    cudaMemcpy(equations, equationsDevice, equations_size, cudaMemcpyDeviceToHost ); 
    cudaMemcpy(&nloops,nloopsDevice,sizeof(int),cudaMemcpyDeviceToHost);
    //printf("c-equations after %.2f \n",equations[0]);

    // free up the memory on the device
    cudaFree(constantsDevice);
    cudaFree(equationsDevice);
#ifdef SIE
    cudaFree(JacobiansDevice);cudaFree(inversesDevice);
#endif
    cudaFree(tendDevice);
    cudaFree(tnowDevice);
    cudaFree(nloopsDevice);

    // return how many steps were taken
    return nloops;
} // cudaIntegrateRK2

extern "C" {
    int WINDIntegrateSystem(
        float tnow, // the current time
        float tend, // the time we integrating the system to
        int n_integration_steps,
        WindFloat * constants, // the constants for each system
        WindFloat * equations, // a flattened array containing the y value for each equation in each system
        int Nsystems, // the number of systems
        int Nequations_per_system, // the number of equations in each system
        float ABSOLUTE, // the absolute tolerance
        float RELATIVE){

        return cudaIntegrateSystem(
            tnow,
            tend,
            n_integration_steps,
            constants,
            equations,
            Nsystems,
            Nequations_per_system,
            ABSOLUTE,
            RELATIVE);
    }
}
