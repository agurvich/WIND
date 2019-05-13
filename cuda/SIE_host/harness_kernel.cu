#include <stdio.h>
#include <cublas_v2.h>

#include "implicit_solver.h"
#include "ode.h"
#include "utils.h"
#include "cuda_utils.h"
#include "vector_kernels.h"
#include "linear_algebra.h"

//#define DEBUG

#define useCUDA

//#include <cusolverDn.h>
//#include "magmablas.h"

void SIE_step(
    float timestep, // device pointer to the current timestep (across all systems, lame!!)
    float ** d_Jacobianss,  // Nsystems x Neqn_p_sys*Neqn_p_sys 2d array with flattened jacobians
    float * d_Jacobianss_flat,
    float ** d_inversess, // inverse output, overwrite d_Jacobianss
    float * d_inversess_flat,
    float ** d_identity, // 1 x Neqn_p_sys*Neqn_p_sys array storing the identity (ideally in constant memory?)
    float ** d_derivatives, // Nsystems x Neqn_p_sys 2d array to store derivatives
    float * d_derivatives_flat, // Nsystems*Neqn_p_sys 1d array (flattened above)
    float * d_equations_flat, // output state vector, iterative calls integrates
    int Nsystems, // number of ODE systems
    int Neqn_p_sys, // number of equations in each system
    float * d_derivative_modification_flat){ // vector to subtract from hf before multipying by A

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

    // scalars for adding/multiplying
    float alpha = 1.0;
    float beta = 0.0;
/* ----------------------------------------------- */


/* -------------- configure the grid  ------------ */
    int threads_per_block; // TODO replace this
    dim3 matrix_gridDim;
    dim3 vector_gridDim;
    dim3 ode_gridDim;

    configureGrid(
        Nsystems,Neqn_p_sys,
        &threads_per_block,
        &matrix_gridDim,
        &ode_gridDim,
        &vector_gridDim);

    
/* ----------------------------------------------- */


/* -------------- invert the matrix -------------- */
    cublasStatus_t error;
    cublasStatus_t second_error;
    if (d_Jacobianss != NULL){
        // compute (I-hJ) with a custom kernel
#ifdef useCUDA
        addArrayToBatchArrays<<<matrix_gridDim,threads_per_block>>>(
            d_identity,d_Jacobianss,1.0,-1.0,timestep,
            Nsystems,Neqn_p_sys); 

        // flush any previous uncaught cuda errors
        cudaError_t cuda_error = cudaGetLastError();
        gjeInvertMatrixBatched<<<Nsystems,threads_per_block>>>(
            d_Jacobianss_flat,
            d_inversess_flat,
            Neqn_p_sys,
            Nsystems);
        //cudaDeviceSynchronize();
#else

        //TODO implement add to batch arrays in C

        
        //TODO implement invert matrix batched in C

#endif

        cudaError_t gjeError = cudaGetLastError();
        if (gjeError != cudaSuccess){
            printf("Inversion failed: %s \n",cudaGetErrorString(gjeError));
        }
    }
/* ----------------------------------------------- */

/* -------------- perform a matrix-vector mult --- */
    if (d_derivative_modification_flat != NULL){
        //  (hf(n)-Delta(n-1)) into d_derivatives_flat
#ifdef useCUDA
        addVectors<<<vector_gridDim,threads_per_block>>>(
            -1.0,d_derivative_modification_flat,
            timestep, d_derivatives_flat,
            d_derivatives_flat,Nsystems,Neqn_p_sys);
#else
        // TODO implement addVectors in C
#endif
    }

    // multiply (I-h*Js)^-1 x fs
    error = cublasSgemmBatched(
        handle,// cublas handle
        CUBLAS_OP_N,// no transformation
        CUBLAS_OP_N,// no transformation
        Neqn_p_sys, //m- number of rows in A (and C)
        1, //n- number of columns in B (and C)
        Neqn_p_sys, //k-number of columns in A and rows in B
        (const float *) &alpha, // alpha scalar
        (const float **) d_inversess, // A matrix
        Neqn_p_sys, // leading dimension of the 2d array storing A??
        (const float **) d_derivatives, // B matrix (or n x 1 column vector)
        Neqn_p_sys, // leading dimension of the 2d array storing B??
        (const float *) &beta, // beta scalar
        (float **) d_derivatives, // output "matrix," let's overwrite B
        Neqn_p_sys, // leading dimension of the 2d array storing C??
        Nsystems); // batch count

        if (error != CUBLAS_STATUS_SUCCESS){
            _cudaGetErrorEnum(error);
            printf("Sgemm broke\n");
        }

/* ----------------------------------------------- */

/* ------------ update the current state --------- */
    if (d_derivative_modification_flat == NULL){
        // scale it explicitly in case calling context needs
        //  h x A(n) x  f(n)
#ifdef useCUDA
        scaleVector<<<vector_gridDim,threads_per_block>>>(
            d_derivatives_flat,
            timestep,
            Nsystems,
            Neqn_p_sys);

        // add ys + h x dys = ys + h x [(I-h*Js)^-1*fs]
        addVectors<<<vector_gridDim,threads_per_block>>>(
            1.0, d_equations_flat,
            1.0, d_derivatives_flat,
            d_equations_flat,Nsystems,Neqn_p_sys);
#else
        // TODO implement scale vector in C
        // TODO implement add vectors in C
#endif

    }
/* ----------------------------------------------- */
    
    // shut down cublas
    cublasDestroy_v2(handle);
    cudaFree(P); cudaFree(INFO);
}

int solveSystem(
    float tnow,
    float tend,
    int n_integration_steps,
    float ** d_Jacobianss, // matrix (jacobian) input
    float * d_Jacobianss_flat,
    float ** d_inversess,
    float * d_inversess_flat,
    float * jacobian_zeros,
    float ** d_identity, // pointer to identity (ideally in constant memory?)
    float ** d_derivatives, // vector (derivatives) input
    float * d_derivatives_flat, // dy vector output
    float * d_current_state_flat, // y vector output
    float * d_previous_delta_flat,
    float * d_constants,
    int Nsystems, // number of systems
    int Neqn_p_sys){

    // make sure we don't overintegrate
    float timestep = (tend-tnow)/n_integration_steps;
    int nsteps = 0; 
/* -------------- configure the grid  ------------ */
    int threads_per_block;
    dim3 vector_gridDim;
    configureGrid(
        Nsystems,Neqn_p_sys,
        &threads_per_block,
        NULL,
        NULL,
        &vector_gridDim);

#ifdef MIDPOINT
    /* ------------- do the special first step ------- */
    // evaluate the derivative and jacobian at 
    //  the current state
    resetSystem(
        d_derivatives,
        d_derivatives_flat,
        d_Jacobianss,
        d_Jacobianss_flat,
        d_constants,
        d_current_state_flat,
        jacobian_zeros,
        Nsystems,
        Neqn_p_sys,
        tnow);

    SIE_step(
        timestep, // Nsystems length vector for timestep to use
        d_Jacobianss,
        d_Jacobianss_flat,
        d_inversess, // inverse output, overwrite d_Jacobianss
        d_inversess_flat,
        d_identity, // pointer to identity (ideally in constant memory?)
        d_derivatives, // vector (derivatives) input
        d_derivatives_flat, // dy vector output-- store h x A(0) x f(0)
        d_current_state_flat, // y vector output
        Nsystems, // number of systems
        Neqn_p_sys,// number of equations in each system
        NULL); // vector to subtract from hf before multipying by A

    // save hA(0)f(0) as Delta(0) for next step
    cudaMemcpy(
        d_previous_delta_flat,
        d_derivatives_flat, // is now hA(n)f(n)
        Nsystems*Neqn_p_sys*sizeof(float),
        cudaMemcpyDeviceToDevice);

    // address special step timestepping issues
    tnow+=timestep;
    n_integration_steps-=1;
    nsteps++;
#endif
/* ----------------------------------------------- */

    cublasHandle_t handle;
    cublasStatus_t error;
    cublasCreate_v2(&handle);
/* -------------- main integration loop ---------- */
    while (nsteps < n_integration_steps){        
        // evaluate the derivative and jacobian at 
        //  the current state
        resetSystem(
            d_derivatives,
            d_derivatives_flat,
            d_Jacobianss,
            d_Jacobianss_flat,
            d_constants,
            d_current_state_flat,
            jacobian_zeros,
            Nsystems,
            Neqn_p_sys,
            tnow);

        SIE_step(
            timestep,
            d_Jacobianss,
            d_Jacobianss_flat,
            d_inversess, // inverse output, overwrite d_Jacobianss
            d_inversess_flat,
            d_identity, // pointer to identity (ideally in constant memory?)
            d_derivatives, // vector (derivatives) input
            d_derivatives_flat, // dy vector output -- store A(n) x (hf(n) - Delta(n-1))
            d_current_state_flat, // y vector output
            Nsystems, // number of systems
            Neqn_p_sys, // number of equations in each system
// flag to change d_equations_flat or just compute A(n) & hA(n)f(n)
#ifndef MIDPOINT
            NULL); // doubles as a flag to add A h f(n) + y(n)
#else
            d_previous_delta_flat);
        
        // add Delta(n) = Delta(n-1) + 2 A x (hf(n) - Delta(n-1))
        //  and overwrite Delta(n-1) with Delta(n) 
        //  now that we don't need the "previous" step
#ifdef useCUDA
        addVectors<<<vector_gridDim,threads_per_block>>>(
            2.0, d_derivatives_flat,
            1.0, d_previous_delta_flat,
            d_previous_delta_flat,Nsystems,Neqn_p_sys);

        // add y(n+1) = y(n) + Delta(n)
        addVectors<<<vector_gridDim,threads_per_block>>>(
            1.0, d_previous_delta_flat, // really the current delta
            1.0, d_current_state_flat,
            d_current_state_flat,Nsystems,Neqn_p_sys);
#else
        // TODO implement add vectors in C
        // TODO implement add vectors in C
#endif

#endif
        tnow+=timestep;
        nsteps++;
    }

#ifdef MIDPOINT
    /* ------------- do the special last step -------- */
    // evaluate the derivative and jacobian at 
    //  the current state
    resetSystem(
        d_derivatives,
        d_derivatives_flat,
        d_Jacobianss,
        d_Jacobianss_flat,
        d_constants,
        d_current_state_flat,
        jacobian_zeros,
        Nsystems,
        Neqn_p_sys,
        tnow);

    SIE_step(
        timestep,
        d_Jacobianss, // matrix (jacobian) input
        d_Jacobianss_flat,
        d_inversess, // inverse output, overwrite d_Jacobianss
        d_inversess_flat,
        d_identity, // pointer to identity (ideally in constant memory?)
        d_derivatives, // vector (derivatives) input
        d_derivatives_flat, // dy vector output -- store A(m) x (hf(m) - Delta(m-1))
        d_current_state_flat, // y vector output
        Nsystems, // number of systems
        Neqn_p_sys, // number of equations in each system
        d_previous_delta_flat); // vector to subtract from hf before multipying by A

#ifdef useCUDA
    // add y(n+1) = y(m) + Delta(m)
    addVectors<<<vector_gridDim,threads_per_block>>>(
        1.0, d_derivatives_flat,
        1.0, d_current_state_flat,
        d_current_state_flat,Nsystems,Neqn_p_sys);
#else
    // TODO implement add vectors in C
#endif    

    // increment tnow and put tend back where we found it
    //  for completeness' sake (even if it doesn't matter)
    tnow+=timestep;
    n_integration_steps+=1;
    nsteps++;
#endif

    cublasDestroy_v2(handle);
    return nsteps;
}

int errorLoop(
    float tnow,
    float tend,
    int n_integration_steps,
    float ** d_Jacobianss, // matrix (jacobian) input
    float * d_Jacobianss_flat,
    float ** d_inversess,
    float * d_inversess_flat,
    float * jacobian_zeros,
    float ** d_identity, // pointer to identity (ideally in constant memory?)
    float ** d_derivatives, // vector (derivatives) input
    float * d_derivatives_flat, // dy vector output
    float * equations,
    float * d_current_state_flat, // y vector output
    float * d_half_current_state_flat,
    float * d_previous_delta_flat,
    float * d_constants,
    int Nsystems, // number of systems
    int Neqn_p_sys){

#ifdef MIDPOINT
    // need at least 3 steps to evaluate midpoint method
    n_integration_steps = max(n_integration_steps,3);
#endif

    int * error_flag = (int *) malloc(sizeof(int));
    int * d_error_flag;
    cudaMalloc(&d_error_flag,sizeof(int));
    *error_flag = 0;
    cudaMemcpy(d_error_flag,error_flag,sizeof(int),cudaMemcpyHostToDevice);

/* -------------- configure the grid  ------------ */
    int threads_per_block;
    dim3 vector_gridDim;
    configureGrid(
        Nsystems,Neqn_p_sys,
        &threads_per_block,
        NULL,
        NULL,
        &vector_gridDim);

    int nsteps=0;

    nsteps+= solveSystem(
            tnow,
            tend,
            n_integration_steps,
            d_Jacobianss,
            d_Jacobianss_flat,
            d_inversess,
            d_inversess_flat,
            jacobian_zeros,
            d_identity,
            d_derivatives,
            d_derivatives_flat,
            d_current_state_flat,
            d_previous_delta_flat,
            d_constants,
            Nsystems,
            Neqn_p_sys);

/* ----------------------------------------------- */
    
#ifdef ADAPTIVETIMESTEP 
    // use a flag as a counter, why not
    int unsolved = 1;
    while (unsolved){
        
        n_integration_steps*=2;
        nsteps+= solveSystem(
            tnow,
            tend,
            n_integration_steps,
            d_Jacobianss,
            d_Jacobianss_flat,
            d_inversess,
            d_inversess_flat,
            jacobian_zeros,
            d_identity,
            d_derivatives,
            d_derivatives_flat,
            d_half_current_state_flat,// the output state vector
            d_previous_delta_flat,
            d_constants,
            Nsystems,
            Neqn_p_sys);

        // determine if ANY of the INDEPENDENT systems are above the 
        //  the tolerance and fail them all. NOTE: this makes them not
        //  independent.
#ifdef useCUDA
        checkError<<<vector_gridDim,threads_per_block>>>(
            d_current_state_flat,d_half_current_state_flat,d_error_flag,
            Nsystems,Neqn_p_sys);

        // copy back the bool flag and determine if we done did it
        cudaMemcpy(error_flag,d_error_flag,sizeof(int),cudaMemcpyDeviceToHost);
        //*error_flag = 0;
#else
    
        // TODO implement check error in C

#endif
        
        if (*error_flag){
            // increase the refinement level
            unsolved++;
            // put an upper limit on the refinement
            if (unsolved > 9){
                break;
            }

#ifdef LOUD
            printf("refining...%d\n",unsolved);
#endif
            *error_flag = 0;

            // reset the error flag on the device
            cudaMemcpy(d_error_flag,error_flag,sizeof(int),cudaMemcpyHostToDevice);
        

            // copy this half-step to the previous full-step to save work
            cudaMemcpy(
                d_current_state_flat,
                d_half_current_state_flat,
                Nsystems*Neqn_p_sys*sizeof(float),
                cudaMemcpyDeviceToDevice);

            // reset the equation for the half-step
            cudaMemcpy(
                d_half_current_state_flat,
                equations,
                Nsystems*Neqn_p_sys*sizeof(float),
                cudaMemcpyHostToDevice);
        }// if unsolved
        else{
            // we did it, let's exit the loop gracefully
            unsolved=0;
        }
    }// while unsolved
#else
        // take only this one step and call it a day, simplest way to 
        //  quit early is to copy the values from d_equations_flat to d_half_equations_flat and
        //  return normally. 
        cudaMemcpy(d_half_current_state_flat,d_current_state_flat,Nsystems*Neqn_p_sys*sizeof(float),cudaMemcpyDeviceToDevice);
#endif
    // free up memory
    cudaFree(d_error_flag);
    free(error_flag);

    // return computations performed
    return nsteps*Nsystems;
}

int cudaIntegrateSIE(
    float tnow, // the current time
    float tend, // the time we integrating the system to
    int n_integration_steps, // the initial timestep to attempt to integrate the system with
    float * constants, // the constants for each system
    float * equations, // a flattened array containing the y value for each equation in each system
    int Nsystems, // the number of systems
    int Neqn_p_sys){ // the number of equations in each system

#ifdef LOUD
#ifdef MIDPOINT
    printf("SIM Received %d systems, %d equations per system\n",Nsystems,Neqn_p_sys);
#else
    printf("SIE Received %d systems, %d equations per system\n",Nsystems,Neqn_p_sys);
#endif
#endif
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

    // allocate memory for Jacobian matrices as a single "batch"
    float *d_inversess_flat;
    float **d_inversess = initializeDeviceMatrix(jacobian_zeros,&d_inversess_flat,Neqn_p_sys*Neqn_p_sys,Nsystems);

    // initialize state-equation vectors
    float * zeros = (float *) malloc(Nsystems*Neqn_p_sys*sizeof(float));
    for (int i=0; i<Neqn_p_sys*Nsystems; i++){
        zeros[i]=0;
    }   

    // constants that define the ODEs
    /* TODO put this in constant memory instead-- does the below work? 
    __constant__ float d_constants[NUM_CONST]; // NUM_CONST #define'd in ode.h
    cudaMemcpyToSymbol(constants,d_constants,sizeof(d_constants));
    */
    float * d_constants;
    cudaMalloc(&d_constants,Nsystems*NUM_CONST*sizeof(float));
    cudaMemcpy(d_constants,constants,Nsystems*NUM_CONST*sizeof(float),cudaMemcpyHostToDevice);

    // state equations, where output will be stored
    float *d_current_state_flat;
    float **d_current_state = initializeDeviceMatrix(equations,&d_current_state_flat,Neqn_p_sys,Nsystems);

    float *d_half_current_state_flat;
    float **d_half_current_state = initializeDeviceMatrix(
        equations,&d_half_current_state_flat,Neqn_p_sys,Nsystems);

#ifdef MIDPOINT
    // saving previous step Y(n-1) because we need that for SIE2
    float *d_previous_delta_flat;
    float **d_previous_delta = initializeDeviceMatrix(zeros,&d_previous_delta_flat,Neqn_p_sys,Nsystems);
#else

    float *d_previous_delta_flat = NULL;

#endif

    // initialize derivative vectors
    float *d_derivatives_flat;
    float **d_derivatives = initializeDeviceMatrix(zeros,&d_derivatives_flat,Neqn_p_sys,Nsystems);

/* ----------------------------------------------- */

    int nsteps = errorLoop(
        tnow,
        tend,
        n_integration_steps,
        d_Jacobianss, // matrix (jacobian) input
        d_Jacobianss_flat,
        d_inversess,
        d_inversess_flat,
        jacobian_zeros,
        d_identity, // pointer to identity (ideally in constant memory?)
        d_derivatives, // vector (derivatives) input
        d_derivatives_flat, // dy vector output
        equations,
        d_current_state_flat, // y vector output
        d_half_current_state_flat,
        d_previous_delta_flat,
        d_constants,
        Nsystems, // number of systems
        Neqn_p_sys);
    
#ifdef LOUD
    printf("nsteps taken: %d - tnow: %.2f\n",nsteps,tend);
#endif

/* -------------- copy data to host -------------- */
    // retrieve the output
    cudaMemcpy(dest, d_half_current_state_flat, Neqn_p_sys*Nsystems*sizeof(float), cudaMemcpyDeviceToHost);
/* ----------------------------------------------- */

/* -------------- shutdown by freeing memory   --- */
    cudaFree(d_identity); cudaFree(d_identity_flat);
    cudaFree(d_Jacobianss); cudaFree(d_Jacobianss_flat);
    cudaFree(d_inversess); cudaFree(d_inversess_flat);
    cudaFree(d_current_state); cudaFree(d_current_state_flat);
    cudaFree(d_half_current_state); cudaFree(d_half_current_state_flat);
#ifdef MIDPOINT
    cudaFree(d_previous_delta); cudaFree(d_previous_delta_flat);
#endif 
    cudaFree(d_derivatives); cudaFree(d_derivatives_flat);

    free(zeros); free(jacobian_zeros);
    free(identity_flat);
/* ----------------------------------------------- */
    //return how many steps were taken
    return nsteps;
}
