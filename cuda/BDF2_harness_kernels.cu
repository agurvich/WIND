// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>
#include <cublas_v2.h>
#include "implicit_solver.h"
//#include <cusolverDn.h>
//#include "magmablas.h"
//#define FIXEDTIMESTEP
//#define COMMENTSIE

void BDF2_step(
    float * p_time, // pointer to current time
    float * d_timestep, // device pointer to the current timestep (across all systems, lame!!)
    float ** d_Jacobianss,  // Nsystems x Neqn_p_sys*Neqn_p_sys 2d array with flattened jacobians
    float ** d_inverse, // Nsystems x Neqn_p_sys*Neqn_p_sys 2d array to store output (same as jacobians to overwrite)
    float ** d_identity, // 1 x Neqn_p_sys*Neqn_p_sys array storing the identity (ideally in constant memory?)
    float ** d_derivatives, // Nsystems x Neqn_p_sys 2d array to store derivatives
    float * d_derivatives_flat, // Nsystems*Neqn_p_sys 1d array (flattened above)
    float * d_out_flat, // output state vector, iterative calls integrates
    int Nsystems, // number of ODE systems
    int Neqn_p_sys){ // number of equations in each system

#ifndef COMMENTSIE
/* -------------- initialize cublas -------------- */
    // initialize cublas status tracking pointers
    cublasHandle_t handle;
    int *P, *INFO;
    // handle is something that connects cublas calls within a stream... something about v2 and 
    // being able to pass scalars by reference instead of by value. I don't really understand it
    // place to store cublas status stuff. 
    cublasCreate_v2(&handle);
    cudaMalloc(&P, Neqn_p_sys * Nsystems * sizeof(int));
    cudaMalloc(&INFO,  Nsystems * sizeof(int));

    cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);
/* ----------------------------------------------- */


/* -------------- calculate the timestep --------- */
    int * d_max_index;
    cudaMalloc(&d_max_index,sizeof(int));
    float * d_alpha, * d_beta;
    cudaMalloc(&d_alpha,sizeof(float));
    cudaMalloc(&d_beta,sizeof(float));

    // scalars for adding/multiplying
    float alpha = 1.0;
    cudaMemcpy(d_alpha,&alpha,sizeof(float),cudaMemcpyHostToDevice);
    float beta = 0.0;
    cudaMemcpy(d_beta,&beta,sizeof(float),cudaMemcpyHostToDevice);

#ifndef FIXEDTIMESTEP
    // TODO don't really understand how this should be working :|
    cublasIsamax(
        handle, // cublas handle
        Nsystems*Neqn_p_sys, // number of elements in the vector
        d_derivatives_flat, // the vector to take the max of
        1, // the stride between elements of the vector
        d_max_index); // the index of the max element of the vector

    // literally just change what the pointer is pointing to on the device
    updateTimestep<<<1,1>>>(d_timestep,d_derivatives_flat,d_max_index);
#endif
/* ----------------------------------------------- */


/* -------------- invert the matrix -------------- */
    // compute (I-hJ) with a custom kernel
    // TODO pretty sure i need a multidimensional grid here, 
    // blocks can't be 160x160 threads
    addArrayToBatchArrays<<<Nsystems,Neqn_p_sys*Neqn_p_sys>>>(d_identity,d_Jacobianss,1.0,-1.0,d_timestep);

    // host call to cublas, does LU factorization for matrices in d_Jacobianss, stores the result in... P?
    // the permutation array seems to be important for some reason
    cublasSgetrfBatched(
        handle, // cublas handle
        Neqn_p_sys, // leading dimension of A??
        d_Jacobianss, // matrix to factor, here I-hs*Js
        Neqn_p_sys, // 
        P, // permutation matrix
        INFO, // cublas status object
        Nsystems); // number of systems

    // second cublas call, this one solves AX=B with B the identity. It puts X in d_inverse
    cublasSgetriBatched(
        handle, // cublas handle
        Neqn_p_sys, // leading dimension of A??
        (const float **)d_Jacobianss, // matrix to inverse, here I-hs*Js
        Neqn_p_sys, // leading dimension of B??
        P, // permutation matrix
        d_inverse, // output matrix
        Neqn_p_sys, // 
        INFO, // cublas status object
        Nsystems); // number of systems
/* ----------------------------------------------- */
    //cudaRoutine<<<1,Neqn_p_sys*Neqn_p_sys>>>(Neqn_p_sys,d_Jacobianss,0);

/* -------------- perform a matrix-vector mult --- */
    // multiply (I-h*Js)^-1 x fs
    cublasSgemmBatched(
        handle,// cublas handle
        CUBLAS_OP_N,// no transformation
        CUBLAS_OP_N,// no transformation
        Neqn_p_sys, //m- number of rows in A (and C)
        1, //n- number of columns in B (and C)
        Neqn_p_sys, //k-number of columns in A and rows in B
        (const float *) d_alpha, // alpha scalar
        (const float **) d_inverse, // A matrix
        Neqn_p_sys, // leading dimension of the 2d array storing A??
        (const float **) d_derivatives, // B matrix (or n x 1 column vector)
        Neqn_p_sys, // leading dimension of the 2d array storing B??
        (const float *) d_beta, // beta scalar
        (float **) d_derivatives, // output "matrix," let's overwrite B
        Neqn_p_sys, // leading dimension of the 2d array storing C??
        Nsystems); // batch count
/* ----------------------------------------------- */

/* -------------- perform a vector addition ------ */
    // scale the dy vectors by the timestep size
    // TODO pretty sure i need a multidimensional grid here, 
    // blocks can't be 160x160 threads
    //scaleVector<<<Nsystems,Neqn_p_sys>>>(d_derivatives_flat,d_timesteps);
    
    //cudaRoutineFlat<<<1,Nsystems*Neqn_p_sys>>>(Neqn_p_sys,d_derivatives_flat);
    // add ys + h x dys = ys + h x [(I-h*Js)^-1*fs]
    /*
    printfCUDA<<<1,1>>>(d_timestep);
    cudaRoutineFlat<<<1,Nsystems*Neqn_p_sys>>>(Neqn_p_sys,d_derivatives_flat);
    cudaRoutineFlat<<<1,Nsystems*Neqn_p_sys>>>(Neqn_p_sys,d_out_flat);
    */
    cublasSaxpy(
        handle, // cublas handle
        Neqn_p_sys*Nsystems, // number of elements in each vector
        (const float *) d_timestep, // alpha scalar <-- can't use device pointer???
        (const float *) d_derivatives_flat, // vector we are adding, flattened derivative vector
        1, // stride between consecutive elements
        d_out_flat, // vector we are replacing
        1); // stride between consecutive elements
    //cudaRoutineFlat<<<1,Nsystems*Neqn_p_sys>>>(Neqn_p_sys,d_out_flat);
/* ----------------------------------------------- */
    
    cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle);
    cudaFree(d_max_index); cudaFree(d_alpha);cudaFree(d_beta);

#endif
    // increment the timestep by whatever we just stepped by
    // allowing the device to vary/choose what it is (so we have
    // to copy it over). In FIXEDTIMESTEP mode this is silly but 
    // even still necessary.
    float timestep = 1.0;
    cudaMemcpy(&timestep,d_timestep,sizeof(float),cudaMemcpyDeviceToHost);
    *p_time+=timestep;

    // shut down cublas
    //TODO should free more stuff here?
}

int cudaIntegrateBDF2(
    float tnow, // the current time
    float tend, // the time we integrating the system to
    float * constants, // the constants for each system
    float * equations, // a flattened array containing the y value for each equation in each system
    int Nsystems, // the number of systems
    int Neqn_p_sys){ // the number of equations in each system

    /*
    int blocksize,gridsize;
    if (Neqn_p_sys*Neqn_p_sys*Nsystems < 1024){
        blocksize = Neqn_p_sys*Neqn_p_sys*Nsystems;
        gridsize = 1;
    }
    else{
        blocksize = 1024;
        gridsize = Neqn_p_sys*Neqn_p_sys*Nsystems/1024+1;
    }

    dim3 dimBlock( blocksize, 1 );
    dim3 dimGrid( gridsize, 1 );
    */

    printf("BDF2 Received %d systems, %d equations per system\n",Nsystems,Neqn_p_sys);
    float *dest = equations;

    // define the identity matrix on the host
    float *identity_flat = (float *)malloc(Neqn_p_sys*Neqn_p_sys*sizeof(float));
    setIdentityMatrix(identity_flat,Neqn_p_sys);
    
    // set a batchsize of one
    float * d_identity_flat;
    float ** d_identity = initializeDeviceMatrix(identity_flat,&d_identity_flat,Neqn_p_sys*Neqn_p_sys,1);
    
/* -------------- move data to device ------------ */
    // zeros to initialize jacobians with
    float * jacobian_zeros = (float *) malloc(Nsystems*Neqn_p_sys*Neqn_p_sys*sizeof(float));
    for (int i=0; i<Neqn_p_sys*Neqn_p_sys*Nsystems; i++){
        jacobian_zeros[i]=0;
    }   

    // allocate memory for Jacobian matrices as a single "batch"
    float *d_Jacobianss_flat;
    float **d_Jacobianss = initializeDeviceMatrix(jacobian_zeros,&d_Jacobianss_flat,Neqn_p_sys*Neqn_p_sys,Nsystems);
    //calculateJacobians<<<Nsystems,Neqn_p_sys>>>(d_Jacobianss,tnow);

    // initialize state-equation vectors
    float * zeros = (float *) malloc(Nsystems*Neqn_p_sys*sizeof(float));
    for (int i=0; i<Neqn_p_sys*Nsystems; i++){
        zeros[i]=0;
    }   
    float *d_out_flat;
    float **d_out = initializeDeviceMatrix(equations,&d_out_flat,Neqn_p_sys,Nsystems);

    // initialize derivative vectors
    float *d_derivatives_flat;
    float **d_derivatives = initializeDeviceMatrix(zeros,&d_derivatives_flat,Neqn_p_sys,Nsystems);

    // initialize single global timestep shared across systems
    float * d_timestep;
    cudaMalloc(&d_timestep,sizeof(float));
    float *temp_timestep=(float *) malloc(sizeof(float));
    *temp_timestep = (tend-tnow)/4.0;
    cudaMemcpy(d_timestep,temp_timestep,sizeof(float),cudaMemcpyHostToDevice);

/* ----------------------------------------------- */

    
/* -------------- main integration loop ---------- */
    int nsteps=0;
    while (tnow < tend){
        nsteps++;
        
        // evaluate the derivative function at tnow
        calculateDerivatives<<<Nsystems,Neqn_p_sys>>>(d_derivatives_flat,tnow);
        //printf("t - %.4f\n",tnow);

        // reset the jacobian, which has been replaced by (I-hJ)^-1
        if (nsteps > 1){
            cudaMemcpy(
                d_Jacobianss_flat,jacobian_zeros,
                Nsystems*Neqn_p_sys*Neqn_p_sys*sizeof(float),
                cudaMemcpyHostToDevice);
            //calculateJacobians<<<Nsystems,Neqn_p_sys>>>(d_Jacobianss,tnow);
        }

        /*
        printfCUDA<<<1,1>>>(d_timestep);
        cudaRoutineFlat<<<1,Nsystems*Neqn_p_sys*Neqn_p_sys>>>(5,d_Jacobianss_flat);
        cudaRoutineFlat<<<1,Nsystems*Neqn_p_sys>>>(5,d_derivatives_flat);
        cudaRoutine<<<1,Neqn_p_sys*Neqn_p_sys>>>(5,d_identity,0);
        cudaRoutineFlat<<<1,Nsystems*Neqn_p_sys>>>(5,d_out_flat);
        */

        //printf("stepping...\n");
        BDF2_step(
            &tnow, //the current time
            d_timestep, // Nsystems length vector for timestep to use
            d_Jacobianss, // matrix (jacobian) input
            d_Jacobianss, // inverse output, overwrite d_Jacobianss
            d_identity, // pointer to identity (ideally in constant memory?)
            d_derivatives, // vector (derivatives) input
            d_derivatives_flat, // dy vector output
            d_out_flat, // y vector output
            Nsystems, // number of systems
            Neqn_p_sys); // number of equations in each system

        //printf("%.2f %.2f\n",tnow,tend);

    }
    printf("nsteps taken: %d\n",nsteps);

/* -------------- copy data to host -------------- */
    // retrieve the output
    cudaMemcpy(dest, d_out_flat, Neqn_p_sys*Nsystems*sizeof(float), cudaMemcpyDeviceToHost);
/* ----------------------------------------------- */

/* -------------- shutdown by freeing memory   --- */
    cudaFree(d_Jacobianss); cudaFree(d_Jacobianss_flat);
    cudaFree(d_out); cudaFree(d_out_flat);
    cudaFree(d_identity); cudaFree(d_identity_flat);
    cudaFree(d_derivatives); cudaFree(d_derivatives_flat);

    free(zeros); free(jacobian_zeros);
    free(temp_timestep);
    free(identity_flat);
/* ----------------------------------------------- */
    //return how many steps were taken
    return nsteps;
}
