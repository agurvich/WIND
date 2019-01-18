
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

void invert(float** src, float** dst, int n, int batchSize)
{
    // handle is something that connects cublas calls within a stream... something about v2 and 
    // being able to pass scalars by reference instead of by value. I don't really understand it
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // place to store cublas status stuff. 
    int *P, *INFO;
    cudaMalloc(&P, n * batchSize * sizeof(int));
    cudaMalloc(&INFO,  batchSize * sizeof(int));

    int lda = n;

    // allocate memory for matrices as a single "batch"
    float **A = (float **)malloc(batchSize*sizeof(float *));
    float **A_d, *A_dflat;
    cudaMalloc(&A_d,batchSize*sizeof(float *));
    cudaMalloc(&A_dflat, n*n*batchSize*sizeof(float));
    A[0] = A_dflat;

    // A holds pointers pointing to different parts of A_dflat corresponding to the 
    // beginning of all the different arrays...  here we fill the pointers in A
    // to be the different parts of memory in A_d
    for (int i = 1; i < batchSize; i++)
      A[i] = A[i-1]+(n*n);
    cudaMemcpy(A_d,A,batchSize*sizeof(float *),cudaMemcpyHostToDevice);

    // copy matrix from src array to flat
    for (int i = 0; i < batchSize; i++)
      cudaMemcpy(A_dflat+(i*n*n), src[i], n*n*sizeof(float), cudaMemcpyHostToDevice);

    // host call to cublas, does LU factorization for matrices in A_d, stores the result in... P? 
    // the permutation array seems to be important for some reason
    // but it is "batching" the call, it's good for inverting a bunch of small matrices where setup
    // could be expensive. Potentially this will be a problem for us? 
    cublasSgetrfBatched(handle,n,A_d,lda,P,INFO,batchSize);

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
    cudaMalloc(&C_dflat, n*n*batchSize*sizeof(float));
    C[0] = C_dflat;
    for (int i = 1; i < batchSize; i++)
      C[i] = C[i-1] + (n*n);

    // second cublas call, this one solves AX=B with B the inverse. It puts X in C_d
    cudaMemcpy(C_d,C,batchSize*sizeof(float *),cudaMemcpyHostToDevice);
    cublasSgetriBatched(handle,n,(const float **)A_d,lda,P,C_d,lda,INFO,batchSize);

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
      cudaMemcpy(dst[i], C_dflat + (i*n*n), n*n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_d); cudaFree(A_dflat); free(A);
    cudaFree(C_d); cudaFree(C_dflat); free(C);
    cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle);
}


void test_invert()
{
    const int n = 3;
    const int mybatch = 4;

    //Random matrix with full pivots
    float full_pivot[n*n] = { 0.5, 3, 4,
                                1, 3, 10,
                                4 , 9, 16 };

    //Almost same as above matrix with first pivot zero
    float zero_pivot[n*n] = { 0, 3, 4,
                              1, 3, 10,
                              4 , 9, 16 };

    float another_zero_pivot[n*n] = { 0, 3, 4,
                                      1, 5, 6,
                                      9, 8, 2 };

    float another_full_pivot[n * n] = { 22, 3, 4,
                                        1, 5, 6,
                                        9, 8, 2 };

    float *result_flat = (float *)malloc(mybatch*n*n*sizeof(float));
    float **results = (float **)malloc(mybatch*sizeof(float *));
    for (int i = 0; i < mybatch; i++)
      results[i] = result_flat + (i*n*n);
    float **inputs = (float **)malloc(mybatch*sizeof(float *));
    inputs[0]  = zero_pivot;
    inputs[1]  = full_pivot;
    inputs[2]  = another_zero_pivot;
    inputs[3]  = another_full_pivot;

    for (int qq = 0; qq < mybatch; qq++){
      fprintf(stdout, "Input %d:\n\n", qq);
      for(int i=0; i<n; i++)
      {
        for(int j=0; j<n; j++)
            fprintf(stdout,"%f\t",inputs[qq][i*n+j]);
        fprintf(stdout,"\n");
      }
    }
    fprintf(stdout,"\n\n");

    invert(inputs, results, n, mybatch);

    for (int qq = 0; qq < mybatch; qq++){
      fprintf(stdout, "Inverse %d:\n\n", qq);
      for(int i=0; i<n; i++)
      {
        for(int j=0; j<n; j++)
            fprintf(stdout,"%f\t",results[qq][i*n+j]);
        fprintf(stdout,"\n");
      }
    }
}

int main()
{
    test_invert();

    return 0;
}
