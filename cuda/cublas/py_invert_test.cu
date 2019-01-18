// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>
#include <cublas_v2.h>
//#include <cusolverDn.h>
  
void printArray(int * arr,int N){
    for (int i = 0; i<N;i++){
        printf("%d ",arr[i]);
    }
    printf("\n");
}

void printFArray(float * arr, int N){
    for (int i = 0; i<N;i++){
        printf("%.2f ",arr[i]);
    }
    printf("\n");
}


__global__ void cudaRoutine(int Narr, float * darr){

    printf("%d - %.2f hello\n",threadIdx.x,darr[threadIdx.x]);

}

void invertMatrix(int batchSize,float * src_flat,int ndim){
    printf("Received %d arrays, each %d x %d:\n",batchSize,ndim,ndim);
    float **src = (float **)malloc(batchSize*sizeof(float *));
    src[0] = src_flat;
    for (int i=1; i<batchSize; i++){
        src[i] = src[i-1]+i*ndim*ndim;
    }

    //float *dest = (float *)malloc(batchSize*ndim*ndim*sizeof(float *));
    float *dest = src_flat;

    int blocksize,gridsize;
    if (ndim*ndim*batchSize < 1024){
        blocksize = ndim*ndim*batchSize;
        gridsize = 1;
    }
    else{
        blocksize = 1024;
        gridsize = ndim*ndim*batchSize/1024+1;
    }

    dim3 dimBlock( blocksize, 1 );
    dim3 dimGrid( gridsize, 1 );

    // handle is something that connects cublas calls within a stream... something about v2 and 
    // being able to pass scalars by reference instead of by value. I don't really understand it
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // place to store cublas status stuff. 
    int *P, *INFO;
    cudaMalloc(&P, ndim * batchSize * sizeof(int));
    cudaMalloc(&INFO,  batchSize * sizeof(int));

    // allocate memory for matrices as a single "batch"
    float **A = (float **)malloc(batchSize*sizeof(float *));
    float **A_d, *A_dflat;
    cudaMalloc(&A_d,batchSize*sizeof(float *));
    cudaMalloc(&A_dflat, ndim*ndim*batchSize*sizeof(float));
    A[0] = A_dflat;

    // A holds pointers pointing to different parts of A_dflat corresponding to the 
    // beginning of all the different arrays...  here we fill the pointers in A
    // to be the different parts of memory in A_d
    for (int i = 1; i < batchSize; i++)
      A[i] = A[i-1]+(ndim*ndim);
    cudaMemcpy(A_d,A,batchSize*sizeof(float *),cudaMemcpyHostToDevice);

    // copy matrix from src array to flat
    for (int i = 0; i < batchSize; i++)
      cudaMemcpy(A_dflat+(i*ndim*ndim), src[i], ndim*ndim*sizeof(float), cudaMemcpyHostToDevice);

    //cudaRoutine<<<dimGrid, dimBlock>>>(ndim*ndim*batchSize,A_dflat);

    // host call to cublas, does LU factorization for matrices in A_d, stores the result in... P? 
    // the permutation array seems to be important for some reason
    // but it is "batching" the call, it's good for inverting a bunch of small matrices where setup
    // could be expensive. Potentially this will be a problem for us? 
    cublasSgetrfBatched(handle,ndim,A_d,ndim,P,INFO,batchSize);

    // copy back info about how the cublas call went
    int INFOh[batchSize];
    cudaMemcpy(INFOh,INFO,batchSize*sizeof(int),cudaMemcpyDeviceToHost);

    // determine if any of the matrix things failed
    for (int i = 0; i < batchSize; i++)
      if(INFOh[i]  != 0)
      {
        fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
      }

    // make a second set of matrices to store the inverses
    float **C = (float **)malloc(batchSize*sizeof(float *));
    float **C_d, *C_dflat;
    cudaMalloc(&C_d,batchSize*sizeof(float *));
    cudaMalloc(&C_dflat, ndim*ndim*batchSize*sizeof(float));
    C[0] = C_dflat;
    for (int i = 1; i < batchSize; i++)
      C[i] = C[i-1] + (ndim*ndim);

    // second cublas call, this one solves AX=B with B the inverse. It puts X in C_d
    cudaMemcpy(C_d,C,batchSize*sizeof(float *),cudaMemcpyHostToDevice);
    cublasSgetriBatched(handle,ndim,(const float **)A_d,ndim,P,C_d,ndim,INFO,batchSize);


    // copy back info from the second cublas call
    cudaMemcpy(INFOh,INFO,batchSize*sizeof(int),cudaMemcpyDeviceToHost);

    // determine if any of those matrices failed
    for (int i = 0; i < batchSize; i++)
      if(INFOh[i] != 0)
      {
        fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
      }

    // copy results to the destination array
    for (int i = 0; i < batchSize; i++)
      cudaMemcpy((void *)(dest+i*ndim*ndim), C_dflat + (i*ndim*ndim), ndim*ndim*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_d); cudaFree(A_dflat); free(A);
    cudaFree(C_d); cudaFree(C_dflat); free(C);
    cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle);

    //printFArray(dest,ndim*ndim*batchSize);
}
