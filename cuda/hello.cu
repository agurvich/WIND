
// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>
#include "python_arradd.h"
 
__global__ 
void hello(int *a, int *b) 
{
    int tid =threadIdx.x+blockDim.x*blockIdx.x;
    a[tid] += b[tid];
}
 
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

int main()
{
    const int Narr = 4; 
    int a[Narr] = {0,1,2,3};
    int b[Narr] = {3,2,1,0};
    int c[Narr] = {0,0,0,0};
    printArray(a,Narr);
    printf("+ \n");
    printArray(b,Narr);
    printf("= \n");

    arradd(Narr,a,b,c);
    printArray(c,Narr);

    return EXIT_SUCCESS;
}
