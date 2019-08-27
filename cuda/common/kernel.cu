#include <stdio.h>
#include <math.h>

#include "config.h"
#include "ode.h"

#ifdef SIE
#include "linear_algebra.h"
#endif

__device__ void checkError(
    WindFloat y1, WindFloat y2,
    int * shared_error_flag,
    float ABSOLUTE, float RELATIVE){
    // determine if any equation is above the absolute or relative tolerances
    WindFloat abs_error = fabs(y2 - y1);
    if(abs_error > ABSOLUTE){
        *shared_error_flag = 1;
#ifdef LOUD
        printf("%d absolute failed: %.2e\n",threadIdx.x,abs_error);
#endif
    }
#if SIE
    WindFloat rel_error = fabs((y2-y1)/(y2+1e-12));
#else
    WindFloat rel_error = fabs((y2-y1)/(2*y2-y1+1e-12));
#endif

    if(rel_error > RELATIVE && 
        fabs(y1) > ABSOLUTE &&
        fabs(y2) > ABSOLUTE){
        *shared_error_flag = 1;
#ifdef LOUD
        printf("%d relative failed: %.2e\n",threadIdx.x,rel_error);
#endif
        }
    __syncthreads();
}

#ifdef SIE
__device__ void  scaleAndInvertJacobians(
    float timestep,
    WindFloat * Jacobians,
    WindFloat * inverses,
    int Nequations_per_system,
    WindFloat * shared_array){

    int this_index;
    // loop through each row and perform 1-hJ
    for (int eqn_i=0; eqn_i<Nequations_per_system; eqn_i++){
        this_index = eqn_i*Nequations_per_system + threadIdx.x;
        Jacobians[this_index] = ((eqn_i)==threadIdx.x) - Jacobians[this_index]*timestep;
    }

    __syncthreads();

    // invert 1-hJ into inverses
    gjeInvertMatrix(
        Jacobians,
        inverses,
        Nequations_per_system,
        shared_array);

    __syncthreads();
}
#endif
    
__device__ WindFloat innerstep(
    float tnow, // the current time
    float tstop, // the time we want to stop
    int n_integration_steps, // the timestep to take
    void * RHS_input, // void pointer to RHS_input struct
    WindFloat * constants, // the constants for each system
    WindFloat * shared_equations, // place to store the current state
    WindFloat * shared_dydts,
    WindFloat * Jacobians, WindFloat * inverses,
    int Nequations_per_system){ // the number of equations in each system

    WindFloat dydt = 0;
    float timestep = (tstop-tnow)/n_integration_steps;

#ifdef SIE
    int this_index;
#endif

    for (int nsteps=0; nsteps<n_integration_steps; nsteps++){
        // limit step size based on remaining time
        timestep = fmin(tstop - tnow, timestep);

        __syncthreads();

        dydt = evaluate_RHS_function(
            tnow,
            RHS_input,
            constants,
            shared_equations,
            shared_dydts,
            Jacobians,
            Nequations_per_system); 
#ifdef SIE

        // invert 1-hJ into inverses
        scaleAndInvertJacobians(
            timestep,
            Jacobians,
            inverses,
            Nequations_per_system,
            shared_dydts);

        // fill the shared array with 
        shared_dydts[threadIdx.x] = dydt;
        __syncthreads();

        /* --  calculate h x (1-hJ)^-1 f  and add it into y(n) -- */
        //  accumulate matrix rows into elements of f
        for (int eqn_i=0; eqn_i < Nequations_per_system; eqn_i++){
            this_index = eqn_i*Nequations_per_system + threadIdx.x;
            // accumulate values directly into shared_equations[eqn_i]-- J and inverses is actualy transposed
            shared_equations[threadIdx.x]+=inverses[this_index]*shared_dydts[eqn_i]*timestep;
        }

#else
        __syncthreads();
        shared_equations[threadIdx.x] += timestep*dydt;

#endif
        tnow+=timestep;
    } // while(tnow < tstop)

    // make sure the last loop finished accumulating
    __syncthreads();
    return shared_equations[threadIdx.x];
}// innerstep

__global__ void integrateSystem(
    float tnow, // the current time
    float tend, // the time we integrating the system to
    float timestep,
    void * RHS_input, // pointer to a RHS_input struct for computing the RHS function
    WindFloat * constants, // the constants for each system
    WindFloat * equations, // a flattened array containing the y value for each equation in each system
    WindFloat * Jacobians,
    WindFloat * inverses,
    int Nsystems, // the number of systems
    int Nequations_per_system,
    int * nsteps, // the number of equations in each system
    float ABSOLUTE, // the absolute tolerance
    float RELATIVE){  // the relative tolerance

    // unique thread ID , based on local ID in block and block ID
    int tid = threadIdx.x + ( blockDim.x * blockIdx.x);

#ifdef SIE
    // offset pointer to find flat jacobian and inverses in global memory
    Jacobians+= Nequations_per_system*Nequations_per_system*blockIdx.x;
    inverses+= Nequations_per_system*Nequations_per_system*blockIdx.x;
#endif

    // offset pointer to find flat constants in global memory
    constants+=NUM_CONST*blockIdx.x;

    extern __shared__ WindFloat total_shared[];
    // total_shared is a pointer to the beginning of this block's shared
    //  memory. If we want to use multiple shared memory arrays we must
    //  manually offset them within that block and allocate enough memory
    //  when initializing the kernel (<<dimGrid,dimBlock,sbytes>>)
    int * shared_error_flag = (int *) &total_shared[0];
    WindFloat * shared_equations = (WindFloat *) &total_shared[1];
    WindFloat * shared_dydts = (WindFloat *) &shared_equations[Nequations_per_system];

    WindFloat y1,y2,current_y;

    int this_nsteps = 0;
    // ensure thread within limit
    int unsolved = 0;
    if (tid < Nsystems*Nequations_per_system ) {
        *shared_error_flag = 0;
        // copy the y values to shared memory
        shared_equations[threadIdx.x] = equations[tid];
        __syncthreads();

        //printf("%d thread %d block\n",threadIdx.x,blockIdx.x);
        while (tnow < tend){
            this_nsteps+=3;

            // make sure we don't overintegrate
            timestep = fmin(tend-tnow,timestep);
            // save this to reset the value before calculating y2
            current_y = shared_equations[threadIdx.x];
            __syncthreads();
            // shared_equations will have the y2 value 
            //  saved in it from the previous loop

            // take the full step
            y1 = innerstep(
                    tnow, tnow+timestep,
                    1,
                    RHS_input,
                    constants,
                    shared_equations,
                    shared_dydts,
                    Jacobians,inverses,
                    Nequations_per_system );

#ifdef DEBUGBLOCK
            if (threadIdx.x==0 && blockIdx.x==DEBUGBLOCK){
                printf("%02d - cuda - y1 ",this_nsteps);
                for (int i=0; i<Nequations_per_system; i++){
                    printf("%.6f\t",shared_equations[i]);
                }
                printf("\n");
            }
#endif
            
            // overwrite the y values in shared memory
            shared_equations[threadIdx.x] = current_y;
            __syncthreads();

            // take the half step
            y2 = innerstep(
                    tnow, tnow+timestep,
                    2,
                    RHS_input,
                    constants,
                    shared_equations,
                    shared_dydts,
                    Jacobians,inverses,
                    Nequations_per_system );

#ifdef DEBUGBLOCK
            if (threadIdx.x==0 && blockIdx.x==DEBUGBLOCK){
                printf("%02d - cuda - y2 ",this_nsteps);
                for (int i=0; i<Nequations_per_system; i++){
                    printf("%.6f\t",shared_equations[i]);
                    }
                printf("\n");
            }

#endif

#ifdef ADAPTIVE_TIMESTEP
            checkError(
                y1,y2,
                shared_error_flag,
                ABSOLUTE,RELATIVE); 
#endif

#ifdef RK2
            if (*shared_error_flag && unsolved <20){
#else
            if (*shared_error_flag && unsolved <10){
#endif

                unsolved++;
                // refine and start over
                timestep/=2;
                *shared_error_flag = 0;
                shared_equations[threadIdx.x] = current_y;
                __syncthreads();
            } // if shared_error_flag
            else{
                unsolved=0;
                // accept this step and update the shared array
#ifdef RK2
                shared_equations[threadIdx.x] = 2*y2-y1;
                __syncthreads();
                // shared_equations already has y2 in it from last
                //  call to innerstep if SIE
#endif
                tnow+=timestep;

#ifdef ADAPTIVE_TIMESTEP
                // let's get a little more optimistic
    #ifdef RK2
                // increase one refinement level
                timestep*=2;
    #else
                // go for gold
                timestep=(tend-tnow);
    #endif

#endif
            }// if shared_error_flag -> else

        }// while tnow < tend

        // copy the y values back to global memory
        equations[tid]=shared_equations[threadIdx.x];
#ifdef LOUD
        if (threadIdx.x == 1 && blockIdx.x == 0){
            printf("nsteps taken: %d - tnow: %.2f\n",this_nsteps,tnow);
        }
#endif

        // accumulate the number of steps this block took
        if (threadIdx.x == 0){
            atomicAdd(nsteps,this_nsteps);
        }
    } // if tid < nequations
} //integrateSystem
