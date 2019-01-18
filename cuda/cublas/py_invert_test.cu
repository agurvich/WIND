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

__global__ void cudaRoutineFlat(int ndim, float * d_arr){
    printf("%d - %.2f hello\n",threadIdx.x,d_arr[threadIdx.x]);
}
__global__ void cudaRoutine(int ndim, float ** d_arr,int index){
    printf("%d - %.2f hello\n",threadIdx.x,d_arr[index][threadIdx.x]);
}

void setIdentityMatrix(float * identity,int ndim){
    for (int i=0; i<(ndim*ndim);i++){
        if (!(i%(ndim+1))){
            identity[i]=1;
        }
        else{
            identity[i]=0;
        }
    }
}

float ** moveFlatBatchArrayToDevice(float * h_flat, int ndim, int nbatch){
    // returns a device pointer to the 2d array, the pointer to the 
    // flat array is the first element of the 2d array, just ask for 
    // more bytes
    
    // allocate device pointers

    float ** d_arr, *d_flat;
    cudaMalloc(&d_arr,nbatch*sizeof(float *));
    cudaMalloc(&d_flat, ndim*ndim*nbatch*sizeof(float));

    // create a temporary array that partitions d_flat
    float **temp = (float **) malloc(nbatch*sizeof(float *));

    // arrange the array in column major order
    temp[0]=d_flat;
    for (int i=1; i<nbatch; i++){
        temp[i]=temp[i-1]+(ndim*ndim);
    }

    // copy the temporary pointer's values to the device
    cudaMemcpy(d_arr,temp,nbatch*sizeof(float *),cudaMemcpyHostToDevice);

    // copy the actual values across
    cudaMemcpy(d_flat,h_flat,ndim*ndim*nbatch*sizeof(float),cudaMemcpyHostToDevice);
    //cudaRoutine<<<1,9>>>(9,d_arr,0);
    return d_arr;
}


__global__ void addArrayToBatchArrays(float ** single_arr, float ** batch_arrs, float alpha, float beta){
    // assumes that gridDim = nbatch and blockDim = ndim
    batch_arrs[blockIdx.x][threadIdx.x]=alpha*single_arr[0][threadIdx.x]+ beta*batch_arrs[blockIdx.x][threadIdx.x];
}

__device__ dotMatrixByVector(float * batch_matrix, float * batch_vector){
    __shared__ new_vector;
    for (int rowi = 0; rowi<ndim;rowi++){
        new_vector[row_i]+=batch_matrix[threadIdx.x]*batch_vector[threadIdx.x];
    }
}

__global__ void multiplyBatchedArraysByBatchedVectors(float ** batch_matrices, float ** batch_vectors,int ndim){
    // move the batched matrix into shared memory
    /*
    extern __shared__ float total_shared[];
    float * s_batch_matrix = (int *) &total_shared[0];
    float * s_vector= (float *) &total_shared[ndim*ndim];

    for (int rowi = 0; rowi<ndim;rowi++){
        s_batch_matrix[threadIdx.x+rowi*ndim]=batch_matrices[blockIdx.x][rowi*ndim+threadIdx.x];
    }

    s_vector[threadIdx.x] = batch_vectors[blockIdx.x][threadIdx.x];
    __syncthreads();
    */
    
    float * batch_matrix = batch_matrices[blockIdx.x];
    float * batch_vectors[blockIdx.x];
    // dot the matrix by the vector
    dotMatrixByVector(batch_matrix,batch_vector);
}

/*float * getDFlatPointer(float ** d_arr,int nbatch){
    float * d_flat;
    float **temp=(float *)malloc(sizeof(float *))
    cudaMemcpy(d_arr,temp,nbatch*sizeof(float *),cudaMemcpyDeviceToHost);
    d_flat = temp[0];
    return d_flat;
    }
*/

void invertMatrix(int batchSize,float * src_flat,int ndim){
    /*
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
    */

    printf("Received %d arrays, each %d x %d:\n",batchSize,ndim,ndim);
    float **src = (float **)malloc(batchSize*sizeof(float *));
    src[0] = src_flat;
    for (int i=1; i<batchSize; i++){
        src[i] = src[i-1]+i*ndim*ndim;
    }

    //float *dest = (float *)malloc(batchSize*ndim*ndim*sizeof(float *));
    float *dest = src_flat;

    // define the identity matrix on the host
    float *identity_flat = (float *)malloc(ndim*ndim*sizeof(float));
    setIdentityMatrix(identity_flat,ndim);
    
    // set a batchsize of one
    float ** d_identity = moveFlatBatchArrayToDevice(identity_flat,ndim,1);
    
    //float *d_identity_flat = *d_identity;
    //cudaRoutineFlat<<<1,9>>>(ndim,d_identity_flat,0);

/* -------------- initialize cublas -------------- */
    // initialize cublas status tracking pointers
    cublasHandle_t handle;
    int *P, *INFO;

    // handle is something that connects cublas calls within a stream... something about v2 and 
    // being able to pass scalars by reference instead of by value. I don't really understand it
    // place to store cublas status stuff. 
    cublasCreate_v2(&handle);
    cudaMalloc(&P, ndim * batchSize * sizeof(int));
    cudaMalloc(&INFO,  batchSize * sizeof(int));
/* ----------------------------------------------- */

/* -------------- move data to device ------------ */
    // allocate memory for matrices as a single "batch"
    float **A_d = moveFlatBatchArrayToDevice(src_flat,ndim,batchSize);
    //float **C_d = moveFlatBatchArrayToDevice(zeros,ndim,batchSize);

    // make a second set of matrices to store the inverses
    float **C = (float **)malloc(batchSize*sizeof(float *));
    float **C_d, *C_dflat;
    cudaMalloc(&C_d,batchSize*sizeof(float *));
    cudaMalloc(&C_dflat, ndim*ndim*batchSize*sizeof(float));
    C[0] = C_dflat;
    for (int i = 1; i < batchSize; i++)
      C[i] = C[i-1] + (ndim*ndim);

    cudaMemcpy(C_d,C,batchSize*sizeof(float *),cudaMemcpyHostToDevice);
/* ----------------------------------------------- */


/* -------------- invert the matrix -------------- */

    // TODO pretty sure i need a multidimensional grid here, 
    // blocks can't be 160x160 threads
    float h = 1.0;
    addArrayToBatchArrays<<<batchSize,ndim*ndim>>>(d_identity,A_d,1.0,-h);
    cudaRoutine<<<1,ndim*ndim>>>(ndim,A_d,0);

    // host call to cublas, does LU factorization for matrices in A_d, stores the result in... P? 
    // the permutation array seems to be important for some reason
    // but it is "batching" the call, it's good for inverting a bunch of small matrices where setup
    // could be expensive. Potentially this will be a problem for us? 
    cublasSgetrfBatched(handle,ndim,A_d,ndim,P,INFO,batchSize);

    // second cublas call, this one solves AX=B with B the inverse. It puts X in C_d
    cublasSgetriBatched(handle,ndim,(const float **)A_d,ndim,P,C_d,ndim,INFO,batchSize);
    //cudaRoutine<<<1,18>>>(ndim,C_d,0);
/* ----------------------------------------------- */

/* -------------- copy data to host -------------- */
    // copy results to the destination array
    for (int i = 0; i < batchSize; i++){
      cudaMemcpy((void *)(dest+i*ndim*ndim), C_dflat + (i*ndim*ndim), ndim*ndim*sizeof(float), cudaMemcpyDeviceToHost);
    }
/* ----------------------------------------------- */

/* -------------- shutdown cublas   -------------- */
    cudaFree(A_d); //cudaFree(A_dflat); free(A);
    cudaFree(C_d); cudaFree(C_dflat); free(C);
    cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle);
/* ----------------------------------------------- */

}
