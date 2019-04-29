#include <stdio.h>
#include <cublas_v2.h>

#include "implicit_solver.h"
#include "ode.h"
#include "utils.h"
#include "cuda_utils.h"
#include "vector_kernels.h"

//#include <cusolverDn.h>
//#include "magmablas.h"

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

/* -------------- initialize cublas -------------- */

    // initialize cublas status tracking pointers
    cublasHandle_t handle;
    int *P, *INFO, *d_INFO_bool;
    int INFO_bool = 0;

    // handle is something that connects cublas calls within a stream... something about v2 and 
    // being able to pass scalars by reference instead of by value. I don't really understand it
    // place to store cublas status stuff. 
    cublasCreate_v2(&handle);
    cudaMalloc(&P, Neqn_p_sys * Nsystems * sizeof(int));
    cudaMalloc(&INFO,  Nsystems * sizeof(int));
    cudaMalloc(&d_INFO_bool,sizeof(int));

    // set the initial value of the INFO_bool check
    cudaMemcpy(&d_INFO_bool,&INFO_bool,sizeof(int),cudaMemcpyHostToDevice);

    //NOTE: uncomment this to use device pointers for constants
    //cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);

    // scalars for adding/multiplying
    float alpha = 1.0;
    float beta = 0.0;
/* ----------------------------------------------- */


/* -------------- configure the grid  ------------ */
    int threads_per_block = min(Neqn_p_sys,MAX_THREADS_PER_BLOCK);
    int x_blocks_per_grid = 1+Neqn_p_sys/MAX_THREADS_PER_BLOCK;
    int y_blocks_per_grid = min(Nsystems,MAX_BLOCKS_PER_GRID);
    int z_blocks_per_grid = 1+Nsystems/MAX_BLOCKS_PER_GRID;

    dim3 matrix_gridDim(
        x_blocks_per_grid*Neqn_p_sys,
        y_blocks_per_grid,
        z_blocks_per_grid);

    dim3 vector_gridDim(
        x_blocks_per_grid,
        y_blocks_per_grid,
        z_blocks_per_grid);

    dim3 ode_gridDim(
        1,
        y_blocks_per_grid,
        z_blocks_per_grid);
/* ----------------------------------------------- */


/* -------------- invert the matrix -------------- */
    // compute (I-hJ) with a custom kernel
    addArrayToBatchArrays<<<matrix_gridDim,threads_per_block>>>(
        d_identity,d_Jacobianss,1.0,-1.0,timestep,
        Nsystems,Neqn_p_sys); 

    // host call to cublas, does LU factorization for matrices in d_Jacobianss, stores the result in... P?
    // the permutation array seems to be important for some reason
    cublasStatus_t error = cublasSgetrfBatched(
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
    //scaleVector<<<vector_gridDim,threads_per_block>>>(d_derivatives_flat,d_timesteps);
    
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
    
    // shut down cublas
    cublasDestroy_v2(handle);
    cudaFree(P); cudaFree(INFO); cudaFree(d_INFO_bool);
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
    float * d_current_state_flat, // y vector output
    float * d_previous_state_flat,
    float ** d_intermediate, // matrix memory for intermediate calculation
    float * d_intermediate_flat,// flattened memory for intermediate calculation
    float * d_constants,
    int Nsystems, // number of systems
    int Neqn_p_sys){

    
    int nsteps = 1; 
/* -------------- configure the grid  ------------ */
    int threads_per_block;
    dim3 vector_gridDim;
    configureGrid(
        Nsystems,Neqn_p_sys,
        &threads_per_block,
        NULL,
        &vector_gridDim,
        NULL);

/* ----------------------------------------------- */
    // copies the values of y(n) -> y(n-1)
    //  now that we don't need the "previous" step
#ifdef ORDER2
    overwriteVector<<<vector_gridDim,threads_per_block>>>(
        d_current_state_flat,
        d_previous_state_flat,
        Nsystems,Neqn_p_sys);
#endif

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
        d_Jacobianss, // matrix (jacobian) input
        d_Jacobianss, // inverse output, overwrite d_Jacobianss
        d_identity, // pointer to identity (ideally in constant memory?)
        d_derivatives, // vector (derivatives) input
        d_derivatives_flat, // dy vector output
        d_current_state_flat, // y vector output
        Nsystems, // number of systems
        Neqn_p_sys); // number of equations in each system

    tnow+=timestep;

/* ----------------------------------------------- */

    cublasHandle_t handle;
    cublasStatus_t error;
    cublasCreate_v2(&handle);
/* -------------- main integration loop ---------- */
    while (tnow < tend){
        nsteps++;
        
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

#ifdef ORDER2
    /* -------------- perform the state switcheroo --- */
        
        //  (y(n)-y(n-1)) into d_intermediate_flat
        addVectors<<<vector_gridDim,threads_per_block>>>(
            -1.0,d_previous_state_flat,
            1.0, d_current_state_flat,
            d_intermediate_flat,Nsystems,Neqn_p_sys);

        // copies the values of y(n) -> y(n-1)
        //  now that we don't need the "previous" step
        overwriteVector<<<vector_gridDim,threads_per_block>>>(
            d_current_state_flat,
            d_previous_state_flat,Nsystems,Neqn_p_sys);
    /* ----------------------------------------------- */
        SIE_step(
            2.0/3.0*timestep, // Nsystems length vector for timestep to use
#else
        SIE_step(
            timestep,
#endif
            d_Jacobianss, // matrix (jacobian) input
            d_Jacobianss, // inverse output, overwrite d_Jacobianss
            d_identity, // pointer to identity (ideally in constant memory?)
            d_derivatives, // vector (derivatives) input
            d_derivatives_flat, // dy vector output
            d_current_state_flat, // y vector output
            Nsystems, // number of systems
            Neqn_p_sys); // number of equations in each system


#ifdef ORDER2
    /* -------------- perform two matrix-vector mults  */
        // multiply (I-2/3h*Js)^-1 x (y(n)-y(n-1)), 
        //  overwrite the output into d_intermediate

        float alpha = 1.0;
        float beta = 0.0;

        error = cublasSgemmBatched(
            handle,// cublas handle
            CUBLAS_OP_N,// no transformation
            CUBLAS_OP_N,// no transformation
            Neqn_p_sys, //m- number of rows in A (and C)
            1, //n- number of columns in B (and C)
            Neqn_p_sys, //k-number of columns in A and rows in B
            (const float *) &alpha, // alpha scalar
            (const float **) d_Jacobianss, // has been replaced by 1-2/3h by most recent SIE_step
            Neqn_p_sys, // leading dimension of the 2d array storing A??
            (const float **) d_intermediate, // B matrix (or n x 1 column vector)
            Neqn_p_sys, // leading dimension of the 2d array storing B??
            (const float *) &beta, // beta scalar
            (float **) d_intermediate, // output "matrix," let's overwrite B
            Neqn_p_sys, // leading dimension of the 2d array storing C??
            Nsystems); // batch count

        addVectors<<<vector_gridDim,threads_per_block>>>(
            1.0/3.0,d_intermediate_flat,
            1.0, d_current_state_flat,
            d_current_state_flat,Nsystems,Neqn_p_sys);

#endif
        tnow+=timestep;

    }
    cublasDestroy_v2(handle);
    return nsteps;
}

int errorLoop(
    float tnow,
    float tend,
    float ** d_Jacobianss, // matrix (jacobian) input
    float * d_Jacobianss_flat,
    float * jacobian_zeros,
    float ** d_identity, // pointer to identity (ideally in constant memory?)
    float ** d_derivatives, // vector (derivatives) input
    float * d_derivatives_flat, // dy vector output
    float * equations,
    float * d_current_state_flat, // y vector output
    float * d_half_current_state_flat,
    float * d_previous_state_flat,
    float ** d_intermediate,
    float * d_intermediate_flat,
    float * d_constants,
    int Nsystems, // number of systems
    int Neqn_p_sys){

    float timestep = tend-tnow;

    // what is our first attempt to solve the system?
#ifdef ORDER2
    int n_integration_steps = 2;
#else
    int n_integration_steps = 1;
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
        &vector_gridDim,
        NULL);

/* ----------------------------------------------- */
    
    // use a flag as a counter, why not
    int unsolved = 1;
    int nsteps=0;
    while (unsolved){
        nsteps+= solveSystem(
            tnow,
            tend,
            timestep/n_integration_steps,
            d_Jacobianss,
            d_Jacobianss_flat,
            jacobian_zeros,
            d_identity,
            d_derivatives,
            d_derivatives_flat,
            d_current_state_flat,
            d_previous_state_flat,
            d_intermediate, // matrix memory for intermediate calculation
            d_intermediate_flat,// flattened memory for intermediate calculation
            d_constants,
            Nsystems,
            Neqn_p_sys);

#ifdef ADAPTIVETIMESTEP 
        n_integration_steps*=2;

        nsteps+= solveSystem(
            tnow,
            tend,
            timestep/n_integration_steps,
            d_Jacobianss,
            d_Jacobianss_flat,
            jacobian_zeros,
            d_identity,
            d_derivatives,
            d_derivatives_flat,
            d_half_current_state_flat,// the output state vector
            d_previous_state_flat,
            d_intermediate, // matrix memory for intermediate calculation
            d_intermediate_flat,// flattened memory for intermediate calculation
            d_constants,
            Nsystems,
            Neqn_p_sys);

        // determine if ANY of the INDEPENDENT systems are above the 
        //  the tolerance and fail them all. NOTE: this makes them not
        //  independent... 
        checkError<<<vector_gridDim,threads_per_block>>>(
            d_current_state_flat,d_half_current_state_flat,d_error_flag,
            Nsystems,Neqn_p_sys);

        // copy back the bool flag and determine if we done did it
        cudaMemcpy(error_flag,d_error_flag,sizeof(int),cudaMemcpyDeviceToHost);
        //*error_flag = 0;
        
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
        
            // reset the equations
            cudaMemcpy(
                d_current_state_flat,
                equations,
                Nsystems*Neqn_p_sys*sizeof(float),
                cudaMemcpyHostToDevice);
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
#else
        // take only this one step and call it a day, simplest way to 
        //  quit early is to copy the values from d_equations_flat to d_half_equations_flat and
        //  return normally. 
        cudaMemcpy(d_half_current_state_flat,d_current_state_flat,Nsystems*Neqn_p_sys*sizeof(float),cudaMemcpyDeviceToDevice);
        unsolved=0;

#endif
    }// while unsolved

    // free up memory
    cudaFree(d_error_flag);
    free(error_flag);

    // return computations performed
    return nsteps;
}

int cudaIntegrateSIE(
    float tnow, // the current time
    float tend, // the time we integrating the system to
    float * constants, // the constants for each system
    float * equations, // a flattened array containing the y value for each equation in each system
    int Nsystems, // the number of systems
    int Neqn_p_sys){ // the number of equations in each system

#ifdef LOUD
#ifdef ORDER2
    printf("SIE2 Received %d systems, %d equations per system\n",Nsystems,Neqn_p_sys);
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
    cudaMalloc(&d_constants,NUM_CONST*sizeof(float));
    cudaMemcpy(d_constants,constants,NUM_CONST*sizeof(float),cudaMemcpyHostToDevice);

    // state equations, where output will be stored
    float *d_current_state_flat;
    float **d_current_state = initializeDeviceMatrix(equations,&d_current_state_flat,Neqn_p_sys,Nsystems);

    float *d_half_current_state_flat;
    float **d_half_current_state = initializeDeviceMatrix(
        equations,&d_half_current_state_flat,Neqn_p_sys,Nsystems);

#ifdef ORDER2
    // saving previous step Y(n-1) because we need that for SIE2
    float *d_previous_state_flat;
    float **d_previous_state = initializeDeviceMatrix(zeros,&d_previous_state_flat,Neqn_p_sys,Nsystems);

    // memory for intermediate calculation... reuse it so we aren't constantly allocating
    //  and deallocating memory, NOTE can we remove this??
    float *d_intermediate_flat;
    float **d_intermediate = initializeDeviceMatrix(zeros,&d_intermediate_flat,Neqn_p_sys,Nsystems);
#else
    float * d_previous_state_flat = NULL;
    float **d_previous_state = NULL;
    float *d_intermediate_flat= NULL;
    float **d_intermediate = NULL;
#endif

    // initialize derivative vectors
    float *d_derivatives_flat;
    float **d_derivatives = initializeDeviceMatrix(zeros,&d_derivatives_flat,Neqn_p_sys,Nsystems);

/* ----------------------------------------------- */

    int nsteps = errorLoop(
        tnow,
        tend,
        d_Jacobianss, // matrix (jacobian) input
        d_Jacobianss_flat,
        jacobian_zeros,
        d_identity, // pointer to identity (ideally in constant memory?)
        d_derivatives, // vector (derivatives) input
        d_derivatives_flat, // dy vector output
        equations,
        d_current_state_flat, // y vector output
        d_half_current_state_flat,
        d_previous_state_flat,
        d_intermediate, // matrix memory for intermediate calculation
        d_intermediate_flat,// flattened memory for intermediate calculation
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
    cudaFree(d_current_state); cudaFree(d_current_state_flat);
    cudaFree(d_half_current_state); cudaFree(d_half_current_state_flat);
#ifdef ORDER2
    cudaFree(d_previous_state); cudaFree(d_previous_state_flat);
    cudaFree(d_intermediate); cudaFree(d_intermediate_flat);
#endif 
    cudaFree(d_derivatives); cudaFree(d_derivatives_flat);

    free(zeros); free(jacobian_zeros);
    free(identity_flat);
/* ----------------------------------------------- */
    //return how many steps were taken
    return nsteps;
}
