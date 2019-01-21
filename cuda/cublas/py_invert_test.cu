// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>
#include <cublas_v2.h>
//#include <cusolverDn.h>


//void find_stepsizes(){}
//void calculate_derivative(){}
//void calculate_Jacobian(){}

//void SIE_step(
//  float * d_hs, //step-sizes to take
//  float ** d_Jacobianss, // a list of flattened jacobians nsystems x (ndim*ndim)
//  float ** d_derivativess, // a list of derivative vectors, nsystems x ndim
//  float ** d_yss, // a list of state vectors, nsystems x ndim
//  float ** d_identity, // the identity matrix, ideally stored in constant memory!!
//  int nsystems, // the number of ODE systems
//  int ndim, // the number of equations in each ODE
//  ){
//  /* Uses the semi-implicit backwards euler method to step multiple systems by h simultaneously:
//      y_n+1 = y_n + h(1 - hJ)^-1 f_n
//      
//  */ 
// -------------- invert the matrix -------------- */

//  // TODO pretty sure i need a multidimensional grid here, 
//  // blocks can't be 160x160 threads
//  // TODO have to have multiple hs here
//  //addArrayToBatchArrays<<<nsystems,ndim*ndim>>>(d_identity,d_Jacobianss,1.0,-h);
//  addArrayToBatchArraysVaryScale<<<nsystems,ndim*ndim>>>(d_identity,d_Jacobianss,-1.0,d_hs);
//  //cudaRoutine<<<1,ndim*ndim>>>(ndim,A_d,0);

//  // host call to cublas, does LU factorization for matrices in A_d, stores the result in... P? 
//  // the permutation array seems to be important for some reason
//  // but it is "batching" the call, it's good for inverting a bunch of small matrices where setup
//  // could be expensive. Potentially this will be a problem for us? 
//  cublasSgetrfBatched(
//      handle, // cublas handle
//      ndim, // leading dimension of matrix ??
//      d_Jacobianss, // matrix to factor
//      ndim, // number of rows&columns in matrix
//      P, // permutation matrix
//      INFO, // cublas status object
//      nsystems); // number of batches

//  // second cublas call, this one solves AX=B with B the inverse. It puts X in C_d
//  cublasSgetriBatched(
//      handle, // cublas handle
//      ndim, // leading dimension of matrix ??
//      (const float **)A_d,
//      ndim,
//      P,
//      C_d,
//      ndim,
//      INFO,
//      nsystems);
// ----------------------------------------------- */

  
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

float ** moveFlatBatchArrayToDevice(float * h_flat, float ** p_d_flat, int arr_size,int nbatch){
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
    //cudaRoutine<<<1,9>>>(9,d_arr,0);

    // return what we want
    *p_d_flat=d_flat;
    return d_arr;
}


__global__ void addArrayToBatchArrays(float ** single_arr, float ** batch_arrs, float alpha, float beta){
    // assumes that gridDim = nbatch and blockDim = ndim
    batch_arrs[blockIdx.x][threadIdx.x]=alpha*single_arr[0][threadIdx.x]+ beta*batch_arrs[blockIdx.x][threadIdx.x];
}

/*float * getDFlatPointer(float ** d_arr,int nbatch){
    float * d_flat;
    float **temp=(float *)malloc(sizeof(float *))
    cudaMemcpy(d_arr,temp,nbatch*sizeof(float *),cudaMemcpyDeviceToHost);
    d_flat = temp[0];
    return d_flat;
    }
*/

void SIE_step(
    float ** A_d, 
    float ** C_d,
    float ** d_identity,
    float ** d_my_vecs,
    float ** d_out,
    int batchSize,
    int ndim){
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

/* -------------- invert the matrix -------------- */

    // TODO pretty sure i need a multidimensional grid here, 
    // blocks can't be 160x160 threads
    float h = 1.0;
    addArrayToBatchArrays<<<batchSize,ndim*ndim>>>(d_identity,A_d,1.0,-h);
    //cudaRoutine<<<1,ndim*ndim>>>(ndim,A_d,0);

    // host call to cublas, does LU factorization for matrices in A_d, stores the result in... P? 
    // the permutation array seems to be important for some reason
    // but it is "batching" the call, it's good for inverting a bunch of small matrices where setup
    // could be expensive. Potentially this will be a problem for us? 
    cublasSgetrfBatched(handle,ndim,A_d,ndim,P,INFO,batchSize);

    // second cublas call, this one solves AX=B with B the inverse. It puts X in C_d
    cublasSgetriBatched(handle,ndim,(const float **)A_d,ndim,P,C_d,ndim,INFO,batchSize);
    //cudaRoutine<<<1,18>>>(ndim,C_d,0);
/* ----------------------------------------------- */

/* -------------- perform a vector mult ---------- */
    
    float alpha = 1.0;
    //float * d_alpha;
    //cudaMalloc(&d_alpha,sizeof(float));
    //cudaMemcpy(d_alpha,&alpha, sizeof(float), cudaMemcpyHostToDevice);

    float beta = 0.0;
    //float * d_beta;
    //cudaMalloc(&d_beta,sizeof(float));
    //cudaMemcpy(d_beta,&beta, sizeof(float), cudaMemcpyHostToDevice);

    // define the identity matrix on the host
    float *many_identity_flat = (float *)malloc(batchSize*ndim*ndim*sizeof(float));
    for (int i=0; i<batchSize; i++){
        setIdentityMatrix(many_identity_flat+ndim*ndim*i,ndim);
    }
    
    // set a batchsize of one
    float * d_many_identity_flat;
    float ** d_many_identity = moveFlatBatchArrayToDevice(many_identity_flat,&d_many_identity_flat,ndim*ndim,batchSize);
    
    addArrayToBatchArrays<<<batchSize,ndim*ndim>>>(d_many_identity,d_many_identity,1.0,1.0);

    cublasSgemmBatched(
        handle,// cublas handle
        CUBLAS_OP_N,// no transformation
        CUBLAS_OP_N,// no transformation
        ndim, //m- number of rows in A (and C)
        1, //n- number of columns in B (and C)
        ndim, //k-number of columns in A and rows in B
        (const float *) &alpha, // alpha scalar
        (const float **) d_many_identity, // A matrix
        ndim, // leading dimension of the 2d array storing A??
        (const float **) d_my_vecs, // B matrix (or n x 1 column vector)
        ndim, // leading dimension of the 2d array storing B??
        (const float *) &beta, // beta scalar
        (float **) d_out, // output "matrix" 
        ndim, // leading dimension of the 2d array storing C??
        batchSize); // batch count
        
    cudaRoutine<<<1,ndim>>>(ndim,d_out,0);
    cudaRoutine<<<1,ndim>>>(ndim,d_out,1);
    
/* ----------------------------------------------- */

    // shut down cublas
    cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle);
}

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
    float * d_identity_flat;
    float ** d_identity = moveFlatBatchArrayToDevice(identity_flat,&d_identity_flat,ndim*ndim,1);
    
/* -------------- move data to device ------------ */
    // allocate memory for matrices as a single "batch"
    float *A_d_flat;
    float **A_d = moveFlatBatchArrayToDevice(src_flat,&A_d_flat,ndim*ndim,batchSize);

    // make a second set of matrices to store the inverses
    float *C_dflat;
    float **C_d = moveFlatBatchArrayToDevice(src_flat,&C_dflat,ndim*ndim,batchSize);


    float * my_vecs = (float *) malloc(batchSize*ndim*sizeof(float));
    for (int i=0; i<ndim*batchSize; i++){
        my_vecs[i]=i;
    }   

    
    float * zeros = (float *) malloc(ndim*sizeof(float));
    for (int i=0; i<ndim; i++){
        zeros[i]=0.0;
    }   

    float *d_my_vecs_flat;
    float **d_my_vecs = moveFlatBatchArrayToDevice(my_vecs,&d_my_vecs_flat,ndim,batchSize);

    float *d_out_flat;
    float **d_out = moveFlatBatchArrayToDevice(zeros,&d_out_flat,ndim,batchSize);

/* ----------------------------------------------- */

    
/* -------------- invert matrices and stuff ------ */
    SIE_step(A_d,C_d,d_identity,d_my_vecs,d_out,batchSize,ndim);
/* ----------------------------------------------- */
    
/* -------------- copy data to host -------------- */
    // copy results to the destination array
    for (int i = 0; i < batchSize; i++){
      cudaMemcpy((void *)(dest+i*ndim*ndim), C_dflat + (i*ndim*ndim), ndim*ndim*sizeof(float), cudaMemcpyDeviceToHost);
    }
/* ----------------------------------------------- */

/* -------------- shutdown cublas   -------------- */
    cudaFree(A_d); //cudaFree(A_dflat); free(A);
    cudaFree(C_d); cudaFree(C_dflat);
/* ----------------------------------------------- */

}
