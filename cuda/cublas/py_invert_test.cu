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

__global__ void cudaRoutineFlat(int neqn, float * d_arr){
    printf("%d - %.2f hello\n",threadIdx.x,d_arr[threadIdx.x]);
}
__global__ void cudaRoutine(int neqn, float ** d_arr,int index){
    printf("%d - %.2f hello\n",threadIdx.x,d_arr[index][threadIdx.x]);
}

void setIdentityMatrix(float * identity,int neqn){
    for (int i=0; i<(neqn*neqn);i++){
        if (!(i%(neqn+1))){
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

    // return what we want
    *p_d_flat=d_flat;
    return d_arr;
}

__global__ void addArrayToBatchArrays(float ** single_arr, float ** batch_arrs, float alpha, float beta, float *betas){
    // assumes that gridDim = nsystems and blockDim = neqn
    batch_arrs[blockIdx.x][threadIdx.x]=alpha*single_arr[0][threadIdx.x]+ beta*betas[blockIdx.x]*batch_arrs[blockIdx.x][threadIdx.x];
}

__global__ void scaleVector(float * vector, float * scales){
    // assumes that gridDim = nsystems and blockDim = neqn
    vector[blockIdx.x*blockDim.x+threadIdx.x]*=scales[blockIdx.x];
}

void SIE_step(
    float * d_timesteps, // nsystems length vector for the timestep for each system
    float ** d_Jacobianss,  // nsystems x neqn*neqn 2d array with flattened jacobians
    float ** d_inverse, // nsystems x neqn*neqn 2d array to store output (can be same as jacobians to overwrite)
    float ** d_identity, // 1 x neqn*neqn array storing the identity (ideally in constant memory?)
    float ** d_derivatives, // nsystems x neqn 2d array to store derivatives
    float * d_derivatives_flat, // nsystems*neqn 1d array (flattened above)
    float * d_out_flat, // output state vector, iterative calls integrates
    int nsystems, // number of ODE systems
    int neqn){ // number of equations in each system

/* -------------- initialize cublas -------------- */
    // initialize cublas status tracking pointers
    cublasHandle_t handle;
    int *P, *INFO;

    // handle is something that connects cublas calls within a stream... something about v2 and 
    // being able to pass scalars by reference instead of by value. I don't really understand it
    // place to store cublas status stuff. 
    cublasCreate_v2(&handle);
    cudaMalloc(&P, neqn * nsystems * sizeof(int));
    cudaMalloc(&INFO,  nsystems * sizeof(int));
/* ----------------------------------------------- */

/* -------------- invert the matrix -------------- */

    // TODO pretty sure i need a multidimensional grid here, 
    // blocks can't be 160x160 threads
    addArrayToBatchArrays<<<nsystems,neqn*neqn>>>(d_identity,d_Jacobianss,1.0,-1.0,d_timesteps);

    // host call to cublas, does LU factorization for matrices in d_Jacobianss, stores the result in... P?
    // the permutation array seems to be important for some reason
    cublasSgetrfBatched(
        handle, // cublas handle
        neqn, // leading dimension of A??
        d_Jacobianss, // matrix to factor, here 1-hs*Js
        neqn, // 
        P, // permutation matrix
        INFO, // cublas status object
        nsystems); // number of systems

    // second cublas call, this one solves AX=B with B the identity. It puts X in d_inverse
    cublasSgetriBatched(
        handle, // cublas handle
        neqn, // leading dimension of A??
        (const float **)d_Jacobianss, // matrix to inverse, here 1-hs*Js
        neqn, // leading dimension of B??
        P, // permutation matrix
        d_inverse, // output matrix
        neqn, // 
        INFO, // cublas status object
        nsystems); // number of systems
/* ----------------------------------------------- */

/* -------------- perform a vector mult ---------- */
    
    // scalars for adding/multiplying
    float alpha = 1.0;
    float beta = 0.0;

    // multiply (1-hs*Js)^-1 x fs
    cublasSgemmBatched(
        handle,// cublas handle
        CUBLAS_OP_N,// no transformation
        CUBLAS_OP_N,// no transformation
        neqn, //m- number of rows in A (and C)
        1, //n- number of columns in B (and C)
        neqn, //k-number of columns in A and rows in B
        (const float *) &alpha, // alpha scalar
        (const float **) d_inverse, // A matrix
        neqn, // leading dimension of the 2d array storing A??
        (const float **) d_derivatives, // B matrix (or n x 1 column vector)
        neqn, // leading dimension of the 2d array storing B??
        (const float *) &beta, // beta scalar
        (float **) d_derivatives, // output "matrix," let's overwrite B
        neqn, // leading dimension of the 2d array storing C??
        nsystems); // batch count
            
/* ----------------------------------------------- */

/* -------------- perform a vector addition ------ */
    // scale the dy vectors by the timestep size
    // TODO pretty sure i need a multidimensional grid here, 
    // blocks can't be 160x160 threads
    scaleVector<<<nsystems,neqn>>>(d_derivatives_flat,d_timesteps);
    
    // add ys + hs*dys = ys + hs*(1-hs*Js)^-1*fs
    cublasSaxpy(
        handle, // cublas handle
        neqn*nsystems, // number of elements in each vector
        (const float *) &alpha, // alpha scalar
        (const float *) d_derivatives_flat, // vector we are adding, flattened derivative vector
        1, // stride between consecutive elements
        d_out_flat, // vector we are replacing
        1); // stride between consecutive elements
/* ----------------------------------------------- */

    cudaRoutineFlat<<<1,neqn*nsystems>>>(neqn,d_out_flat);

    // shut down cublas
    cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle);
}

void invertMatrix(int nsystems,float * src_flat,int neqn){
    /*
    int blocksize,gridsize;
    if (neqn*neqn*nsystems < 1024){
        blocksize = neqn*neqn*nsystems;
        gridsize = 1;
    }
    else{
        blocksize = 1024;
        gridsize = neqn*neqn*nsystems/1024+1;
    }

    dim3 dimBlock( blocksize, 1 );
    dim3 dimGrid( gridsize, 1 );
    */

    printf("Received %d arrays, each %d x %d:\n",nsystems,neqn,neqn);
    float **src = (float **)malloc(nsystems*sizeof(float *));
    src[0] = src_flat;
    for (int i=1; i<nsystems; i++){
        src[i] = src[i-1]+i*neqn*neqn;
    }

    //float *dest = (float *)malloc(nsystems*neqn*neqn*sizeof(float *));
    float *dest = src_flat;

    // define the identity matrix on the host
    float *identity_flat = (float *)malloc(neqn*neqn*sizeof(float));
    setIdentityMatrix(identity_flat,neqn);
    
    // set a batchsize of one
    float * d_identity_flat;
    float ** d_identity = initializeDeviceMatrix(identity_flat,&d_identity_flat,neqn*neqn,1);
    
/* -------------- move data to device ------------ */
    // allocate memory for matrices as a single "batch"
    float *d_Jacobianss_flat;
    float **d_Jacobianss = initializeDeviceMatrix(src_flat,&d_Jacobianss_flat,neqn*neqn,nsystems);

    float * my_vecs = (float *) malloc(nsystems*neqn*sizeof(float));

    for (int i=0; i<neqn*nsystems; i++){
        my_vecs[i]=i;
    }   

    // input derivative vectors
    float *d_derivatives_flat;
    float **d_derivatives = initializeDeviceMatrix(my_vecs,&d_derivatives_flat,neqn,nsystems);

    // initialize state vectors
    float * zeros = (float *) malloc(nsystems*neqn*sizeof(float));
    for (int i=0; i<neqn*nsystems; i++){
        zeros[i]=0;
    }   

    float *d_out_flat;
    float **d_out = initializeDeviceMatrix(zeros,&d_out_flat,neqn,nsystems);

    // timesteps
    float * timesteps_init = (float *) malloc(nsystems*sizeof(float));
    for (int i=0; i<nsystems; i++){
        timesteps_init[i]=i+1;
    }   

    float * d_timesteps;
    cudaMalloc(&d_timesteps,nsystems*sizeof(float));
    cudaMemcpy(d_timesteps,timesteps_init,nsystems*sizeof(float),cudaMemcpyHostToDevice);
/* ----------------------------------------------- */

    
/* -------------- main integration loop ---------- */
    int nsteps = 0;
    while (nsteps < 1){
/*
        if (nstep != 0){
            calculateDerivatives(d_derivatives_flat);
            getTimesteps(d_derivatives_flat,d_timesteps);
            calculateJacobians(d_Jacobianss);
        }
*/

        SIE_step(
            d_timesteps, // nsystems length vector for timestep to use
            d_Jacobianss, // matrix (jacobian) input
            d_Jacobianss, // inverse output, overwrite d_Jacobianss
            d_identity, // pointer to identity (ideally in constant memory?)
            d_derivatives, // vector (derivatives) input
            d_derivatives_flat, // dy vector output
            d_out_flat, // y vector output
            nsystems, // number of systems
            neqn); // number of equations in each system
        nsteps++;
    }
/* ----------------------------------------------- */
    
/* -------------- copy data to host -------------- */
    // copy results to the destination array
    for (int i = 0; i < nsystems; i++){
      cudaMemcpy((void *)(dest+i*neqn*neqn), d_Jacobianss_flat + (i*neqn*neqn), neqn*neqn*sizeof(float), cudaMemcpyDeviceToHost);
    }
/* ----------------------------------------------- */

/* -------------- shutdown cublas   -------------- */
    cudaFree(d_Jacobianss); //cudaFree(d_Jacobianssflat); free(A);
/* ----------------------------------------------- */

}
