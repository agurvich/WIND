 #include <stdio.h>
#include "utils.h"

void GDBbreakpoint(){
    ; // do nothing
}

void printFArray(float * arr, int N){
    for (int i = 0; i<N;i++){
        printf("%.2f ",arr[i]);
    }
    printf("\n");
}

void setIdentityMatrix(float * identity,int Neqn_p_sys){
    for (int i=0; i<(Neqn_p_sys*Neqn_p_sys);i++){
        if (!(i%(Neqn_p_sys+1))){
            identity[i]=1;
        }
        else{
            identity[i]=0;
        }
    }
}

float ** initializeDeviceMatrix(float * h_flat, float ** p_d_flat, int arr_size,int nbatch){
    // returns a device pointer to the 2d array, the pointer to the 
    // flat array is the first element of the 2d array, just ask for 
    // more bytes
    
    // allocate device pointers

    float ** d_arr, *d_flat;
    cudaMalloc(&d_arr,nbatch*sizeof(float *));
    cudaMalloc(&d_flat, arr_size*nbatch*sizeof(float));

    // create a temporary array that partitions d_flat
    float **temp = (float **) malloc(nbatch*sizeof(float *));

    // arrange the array in column major order
    temp[0]=d_flat;
    for (int i=1; i<nbatch; i++){
        temp[i]=temp[i-1]+(arr_size);
    }

    // copy the temporary pointer's values to the device
    cudaMemcpy(d_arr,temp,nbatch*sizeof(float *),cudaMemcpyHostToDevice);
    // copy the actual values across
    cudaMemcpy(d_flat,h_flat,arr_size*nbatch*sizeof(float),cudaMemcpyHostToDevice);

    free(temp);
    // return what we want
    *p_d_flat=d_flat;
    return d_arr;
}
