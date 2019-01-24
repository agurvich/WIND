
// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>
#include "explicit_solver.h"
  
void printArray(int * arr,int N){
    for (int i = 0; i<N;i++){
        printf("%d ",arr[i]);
    }
    printf("\n");
}

void arradd(int Narr,int * arr_a,int * arr_b,int * arr_c){
    printf("Received %d:\n",Narr);
    printArray(arr_a,Narr);
    printArray(arr_b,Narr);
    int *ad;
    int *bd;
    const int isize = Narr*sizeof(int);
 
    cudaMalloc( (void**)&ad, isize ); 
    cudaMalloc( (void**)&bd, isize ); 
    cudaMemcpy( ad, arr_a, isize, cudaMemcpyHostToDevice ); 
    cudaMemcpy( bd, arr_b, isize, cudaMemcpyHostToDevice ); 
    
    int blocksize,gridsize;
    if (Narr < THREAD_BLOCK_LIMIT){
        blocksize = Narr;
        gridsize = 1;
    }
    else{
        blocksize = THREAD_BLOCK_LIMIT;
        gridsize = Narr/THREAD_BLOCK_LIMIT+1;
    }

    dim3 dimBlock( blocksize, 1 );
    dim3 dimGrid( gridsize, 1 );
    hello<<<dimGrid, dimBlock>>>(ad, bd);
    cudaMemcpy( arr_c, ad, isize, cudaMemcpyDeviceToHost ); 

    printf("Returning:\n");
    printArray(arr_c,Narr);

    cudaFree( ad );
    cudaFree( bd );
}


int bar()
{
    const int Narr = 4; 
    int a[Narr] = {0,1,2,3};
    int b[Narr] = {3,2,1,0};
    int * c;
    c = (int *) malloc(sizeof(int)*Narr);

    printArray(a,Narr);
    printf("+ \n");
    printArray(b,Narr);
    printf("= \n");

    arradd(Narr,a,b,c);
    printArray(c,Narr);
    return 1;
}

void cudaIntegrateEuler(
    float tnow, // the current time
    float tend, // the time we integrating the system to
    float * constants, // the constants for each system
    float * equations, // a flattened array containing the y value for each equation in each system
    int Nsystems, // the number of systems
    int Nequations_per_system){ // the number of equations in each system

    printf("Forward Euler Received %d systems, %d equations per system\n",Nsystems,Nequations_per_system);

    // copy the arrays over to the device
    int Nequations = Nsystems*Nequations_per_system;
    int equations_size = Nequations*sizeof(float);
    float *constantsDevice;
    float *equationsDevice;
    cudaMalloc((void**)&constantsDevice, equations_size); 
    cudaMalloc((void**)&equationsDevice, equations_size); 
    cudaMemcpy( constantsDevice, constants, equations_size, cudaMemcpyHostToDevice ); 
    cudaMemcpy( equationsDevice, equations, equations_size, cudaMemcpyHostToDevice ); 

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

    //bar();
    integrate_euler <<<dimGrid,dimBlock,
        2*Nequations_per_system*sizeof(float)+ sizeof(int)
         >>> (
        tnow, tend,
        constantsDevice,equationsDevice,
        Nsystems,Nequations_per_system);
    
    // copy the new state back
    cudaMemcpy(equations, equationsDevice, equations_size, cudaMemcpyDeviceToHost ); 
    //printf("c-equations after %.2f \n",equations[0]);

    // free up the memory on the device
    cudaFree(constantsDevice);
    cudaFree(equationsDevice);

} // cudaIntegrateEuler

