#include <stdio.h>
#include <cublas_v2.h>

#include "implicit_solver.h"
#include "ode.h"
#include "utils.h"
#include "cuda_utils.h"
#include "vector_kernels.h"

//#include <cusolverDn.h>
//#include "magmablas.h"

//#define COMMENTSIE

void SIE_step(
    float timestep, // device pointer to the current timestep (across all systems, lame!!)
    float ** d_Jacobianss,  // Nsystems x Neqn_p_sys*Neqn_p_sys 2d array with flattened jacobians
    float ** d_inverse, // Nsystems x Neqn_p_sys*Neqn_p_sys 2d array to store output (same as jacobians to overwrite)
    float ** d_identity, // 1 x Neqn_p_sys*Neqn_p_sys array storing the identity (ideally in constant memory?)
    float ** d_derivatives, // Nsystems x Neqn_p_sys 2d array to store derivatives
    float * d_derivatives_flat, // Nsystems*Neqn_p_sys 1d array (flattened above)
    float * d_equations_flat, // output state vector, iterative calls integrates
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

    //NOTE: uncomment this to use device pointers for constants
    //cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);
/* ----------------------------------------------- */


/* -------------- calculate the timestep --------- */
    /*
    float * d_alpha, * d_beta;
    cudaMalloc(&d_alpha,sizeof(float));
    cudaMalloc(&d_beta,sizeof(float));
    cudaMemcpy(d_alpha,&alpha,sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,&beta,sizeof(float),cudaMemcpyHostToDevice);
    */

    // scalars for adding/multiplying
    float alpha = 1.0;
    float beta = 0.0;
/* ----------------------------------------------- */


/* -------------- configure the grid  ------------ */
    int threads_per_block = min(Neqn_p_sys,MAX_THREADS_PER_BLOCK);
    int x_blocks_per_grid = 1+Neqn_p_sys/MAX_THREADS_PER_BLOCK;
    int y_blocks_per_grid = Nsystems;

    dim3 matrix_gridDim(
        x_blocks_per_grid*Neqn_p_sys,
        y_blocks_per_grid);
    dim3 vector_gridDim(x_blocks_per_grid,y_blocks_per_grid);
    dim3 blockDim(threads_per_block);
/* ----------------------------------------------- */


/* -------------- invert the matrix -------------- */
    // compute (I-hJ) with a custom kernel
    addArrayToBatchArrays<<<matrix_gridDim,blockDim>>>(
        d_identity,d_Jacobianss,1.0,-1.0,timestep,
        Neqn_p_sys); 

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
        (const float *) &alpha, // alpha scalar
        (const float **) d_inverse, // A matrix
        Neqn_p_sys, // leading dimension of the 2d array storing A??
        (const float **) d_derivatives, // B matrix (or n x 1 column vector)
        Neqn_p_sys, // leading dimension of the 2d array storing B??
        (const float *) &beta, // beta scalar
        (float **) d_derivatives, // output "matrix," let's overwrite B
        Neqn_p_sys, // leading dimension of the 2d array storing C??
        Nsystems); // batch count
/* ----------------------------------------------- */

/* -------------- perform a vector addition ------ */
    // scale the dy vectors by the timestep size
    //scaleVector<<<vector_gridDim,blockDim>>>(d_derivatives_flat,d_timesteps);
    
    // add ys + h x dys = ys + h x [(I-h*Js)^-1*fs]
    cublasSaxpy(
        handle, // cublas handle
        Neqn_p_sys*Nsystems, // number of elements in each vector
        (const float *) &timestep, // alpha scalar <-- can't use device pointer???
        (const float *) d_derivatives_flat, // vector we are adding, flattened derivative vector
        1, // stride between consecutive elements
        d_equations_flat, // vector we are replacing
        1); // stride between consecutive elements
/* ----------------------------------------------- */
    
    cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle);
    //cudaFree(d_max_index); cudaFree(d_alpha);cudaFree(d_beta);

#endif
    // increment the timestep by whatever we just stepped by
    // allowing the device to vary/choose what it is (so we have
    // to copy it over). In FIXEDTIMESTEP mode this is silly but 
    // even still necessary.
    //float timestep = 1.0;
    //cudaMemcpy(&timestep,d_timestep,sizeof(float),cudaMemcpyDeviceToHost);
    //*p_time+=*timestep;

    // shut down cublas
}

int solveSystem(
    float tnow,
    float tend,
    float timestep, 
    float ** d_Jacobianss, // matrix (jacobian) input
    float * d_Jacobianss_flat,
    float * jacobian_zeros,
    float ** d_identity, // pointer to identity (ideally in constant memory?)
    float ** d_derivatives, // vector (derivatives) input
    float * d_derivatives_flat, // dy vector output
    float * d_equations_flat, // y vector output
    float * d_constants,
    int Nsystems, // number of systems
    int Neqn_p_sys){ // number of equations in each system

/* -------------- main integration loop ---------- */
    int nsteps=0;
    while (tnow < tend){
        nsteps++;
        /* ------- reset the derivatives and jacobian matrix ------ */
        // evaluate the derivative function at tnow
        calculateDerivatives<<<Nsystems,1>>>(d_derivatives_flat,d_constants,d_equations_flat,Neqn_p_sys,tnow);

        cudaMemcpy(
            d_Jacobianss_flat,jacobian_zeros,
            Nsystems*Neqn_p_sys*Neqn_p_sys*sizeof(float),
            cudaMemcpyHostToDevice);

        calculateJacobians<<<Nsystems,1>>>(d_Jacobianss,d_constants,d_equations_flat,Neqn_p_sys,tnow);

        //printf("stepping...\n");
        SIE_step(
            timestep, // Nsystems length vector for timestep to use
            d_Jacobianss, // matrix (jacobian) input
            d_Jacobianss, // inverse output, overwrite d_Jacobianss
            d_identity, // pointer to identity (ideally in constant memory?)
            d_derivatives, // vector (derivatives) input
            d_derivatives_flat, // dy vector output
            d_equations_flat, // y vector output
            Nsystems, // number of systems
            Neqn_p_sys); // number of equations in each system

        //printf("%.2f %.2f\n",tnow,tend);
        tnow+=timestep;

    }
    //printf("nsteps taken: %d - tnow: %.2f\n",nsteps,tnow);
    return nsteps;
}

int SIEErrorLoop(
    float tnow,
    float tend,
    float ** d_Jacobianss, // matrix (jacobian) input
    float * d_Jacobianss_flat,
    float * jacobian_zeros,
    float ** d_identity, // pointer to identity (ideally in constant memory?)
    float ** d_derivatives, // vector (derivatives) input
    float * d_derivatives_flat, // dy vector output
    float * equations,
    float * d_equations_flat, // y vector output
    float * d_half_equations_flat,
    float * d_constants,
    int Nsystems, // number of systems
    int Neqn_p_sys){

    float timestep = (tend-tnow);

    int * error_flag = (int *) malloc(sizeof(int));
    int * d_error_flag;
    cudaMalloc(&d_error_flag,sizeof(int));
    *error_flag = 0;
    cudaMemcpy(d_error_flag,error_flag,sizeof(int),cudaMemcpyHostToDevice);
    
    // use a flag as a counter, why not
    int unsolved = 1;
    int nsteps=0;
    //*timestep=0.125;
    while (unsolved){
        nsteps+= solveSystem(
            tnow,
            tend,
            timestep,
            d_Jacobianss,
            d_Jacobianss_flat,
            jacobian_zeros,
            d_identity,
            d_derivatives,
            d_derivatives_flat,
            d_equations_flat,
            d_constants,
            Nsystems,
            Neqn_p_sys);

        timestep/=2.0;

        nsteps+= solveSystem(
            tnow,
            tend,
            timestep,
            d_Jacobianss,
            d_Jacobianss_flat,
            jacobian_zeros,
            d_identity,
            d_derivatives,
            d_derivatives_flat,
            d_half_equations_flat,
            d_constants,
            Nsystems,
            Neqn_p_sys);

        /* -------------- configure the grid  ------------ */
        int threads_per_block = min(Neqn_p_sys,MAX_THREADS_PER_BLOCK);
        int x_blocks_per_grid = 1+Neqn_p_sys/MAX_THREADS_PER_BLOCK;
        int y_blocks_per_grid = Nsystems;

        dim3 vector_gridDim(x_blocks_per_grid,y_blocks_per_grid);
        dim3 blockDim(threads_per_block);
        /* ----------------------------------------------- */

        checkError<<<vector_gridDim,threads_per_block>>>(
            d_equations_flat,d_half_equations_flat,d_error_flag,
            Nsystems,Neqn_p_sys);

        // copy back the bool flag and determine if we done did it
        cudaMemcpy(error_flag,d_error_flag,sizeof(int),cudaMemcpyDeviceToHost);
        //*error_flag = 0;
        if (unsolved > 15){
            break;
        }
        if (*error_flag){
            //printf("refining...%d\n",unsolved);
            *error_flag = 0;
            cudaMemcpy(d_error_flag,error_flag,sizeof(int),cudaMemcpyHostToDevice);
            unsolved++;
            //printf("new timestep: %.2e\n",timestep);
            // reset the equations
            cudaMemcpy(d_equations_flat,equations,Nsystems*Neqn_p_sys*sizeof(float),cudaMemcpyHostToDevice);
            cudaMemcpy(d_half_equations_flat,equations,Nsystems*Neqn_p_sys*sizeof(float),cudaMemcpyHostToDevice);
        }
        else{
            unsolved=0;
        }

    }// while unsolved
    return nsteps;
}


int cudaIntegrateSIE(
    float tnow, // the current time
    float tend, // the time we integrating the system to
    float * constants, // the constants for each system // TODO const? 
    float * equations, // a flattened array containing the y value for each equation in each system
    int Nsystems, // the number of systems
    int Neqn_p_sys// the number of equations in each system
    ){ 

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

    printf("SIE Received %d systems, %d equations per system\n",Nsystems,Neqn_p_sys);
    int threads_per_block = min(Neqn_p_sys,MAX_THREADS_PER_BLOCK);
    int x_blocks_per_grid = 1+Neqn_p_sys/MAX_THREADS_PER_BLOCK;
    int y_blocks_per_grid = Nsystems;
    printf("yb: %d xb*neps: %d tpb: %d\n",y_blocks_per_grid,x_blocks_per_grid*Neqn_p_sys,threads_per_block);

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

    // initialize state-equation vectors
    float * zeros = (float *) malloc(Nsystems*Neqn_p_sys*sizeof(float));
    for (int i=0; i<Neqn_p_sys*Nsystems; i++){
        zeros[i]=0;
    }   
    float *d_equations_flat;
    float **d_equations = initializeDeviceMatrix(equations,&d_equations_flat,Neqn_p_sys,Nsystems);

    float *d_half_equations_flat;
    float **d_half_equations = initializeDeviceMatrix(equations,&d_half_equations_flat,Neqn_p_sys,Nsystems);

    // initialize derivative vectors
    float *d_derivatives_flat;
    float **d_derivatives = initializeDeviceMatrix(zeros,&d_derivatives_flat,Neqn_p_sys,Nsystems);


    // constants that define the ODEs
    /* TODO put this in constant memory instead-- does the below work? 
    __constant__ float d_constants[NUM_CONST]; // NUM_CONST #define'd in ode.h
    cudaMemcpyToSymbol(constants,d_constants,sizeof(d_constants));
    */
    float * d_constants;
    cudaMalloc(&d_constants,NUM_CONST*sizeof(float));
    cudaMemcpy(d_constants,constants,NUM_CONST*sizeof(float),cudaMemcpyHostToDevice);

    // initialize single global timestep shared across systems
    //float * d_timestep;
    //cudaMalloc(&d_timestep,sizeof(float));
    //float *temp_timestep=(float *) malloc(sizeof(float));
    //*temp_timestep = (tend-tnow)/4.0;
    //cudaMemcpy(d_timestep,temp_timestep,sizeof(float),cudaMemcpyHostToDevice);

    int nsteps = SIEErrorLoop(
        tnow,
        tend,
        d_Jacobianss, // matrix (jacobian) input
        d_Jacobianss_flat,
        jacobian_zeros,
        d_identity, // pointer to identity (ideally in constant memory?)
        d_derivatives, // vector (derivatives) input
        d_derivatives_flat, // dy vector output
        equations,
        d_equations_flat, // y vector output
        d_half_equations_flat,
        d_constants,
        Nsystems, // number of systems
        Neqn_p_sys);

/* ----------------------------------------------- */

/* -------------- copy data to host -------------- */
    // retrieve the output
    cudaMemcpy(dest, d_half_equations_flat, Neqn_p_sys*Nsystems*sizeof(float), cudaMemcpyDeviceToHost);
/* ----------------------------------------------- */

/* -------------- shutdown by freeing memory   --- */
    cudaFree(d_Jacobianss); cudaFree(d_Jacobianss_flat);
    cudaFree(d_equations); cudaFree(d_equations_flat);
    cudaFree(d_identity); cudaFree(d_identity_flat);
    cudaFree(d_derivatives); cudaFree(d_derivatives_flat);

    free(zeros); free(jacobian_zeros);
    //free(temp_timestep);
    free(identity_flat);
/* ----------------------------------------------- */
    //return how many steps were taken
    printf("nsteps taken: %d - tnow: %.2f\n",nsteps,tend);
    return nsteps;
}
