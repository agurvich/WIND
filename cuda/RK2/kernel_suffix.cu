__device__ float rk2_innerstep(
    float tnow, // the current time
    float tstop, // the time we want to stop
    float timestep, // the timestep to take
    float * constants, // the constants for each system
    float * shared_temp_equations, // place to store temporary equations
    int Nsystems, // the number of systems
    int Nequations_per_system){ // the number of equations in each system

    float dydt = 0;

    while (tnow < tstop){
        // limit step size based on remaining time
        timestep = fmin(tstop - tnow, timestep);

        //calculate the derivative for this equation
        dydt = calculate_dydt(
            tnow,
            constants,
            shared_temp_equations);

        // update value of temporary equations
        shared_temp_equations[threadIdx.x] += timestep*dydt;
        tnow+=timestep;

    } // while(tnow < tstop)
    return shared_temp_equations[threadIdx.x];
}// rk2_innerstep

__global__ void integrate_rk2(
    float tnow, // the current time
    float tend, // the time we integrating the system to
    float timestep,
    float * constants, // the constants for each system
    float * equations, // a flattened array containing the y value for each equation in each system
    int Nsystems, // the number of systems
    int Nequations_per_system,
    int * nsteps){ // the number of equations in each system

    // unique thread ID , based on local ID in block and block ID
    int tid = threadIdx.x + ( blockDim.x * blockIdx.x);

    extern __shared__ float total_shared[];
    // total_shared is a pointer to the beginning of this block's shared
    //  memory. If we want to use multiple shared memory arrays we must
    //  manually offset them within that block and allocate enough memory
    //  when initializing the kernel (<<dimGrid,dimBlock,sbytes>>)
    int * shared_error_flag = (int *) &total_shared[0];
    float * shared_equations = (float *) &total_shared[1];
    float * shared_temp_equations = (float *) &shared_equations[Nequations_per_system];

    float y1,y2;

    // ensure thread within limit
    if (tid < Nsystems*Nequations_per_system ) {
        // copy the y values to shared memory
        shared_equations[threadIdx.x] = equations[tid];
        *shared_error_flag = 0;
        __syncthreads();

        //printf("%d thread %d block\n",threadIdx.x,blockIdx.x);
        while (tnow < tend){
            // make sure we don't overintegrate
            timestep = fmin(tend-tnow,timestep);

            // now reset the temporary equations
            shared_temp_equations[threadIdx.x] = shared_equations[threadIdx.x];
            __syncthreads();

            y1 = rk2_innerstep(
                tnow, tnow+timestep,
                timestep,
                constants,
                shared_temp_equations,
                Nsystems, Nequations_per_system );
            
            // now reset the temporary equations
            shared_temp_equations[threadIdx.x] = shared_equations[threadIdx.x];
            __syncthreads();

            y2 = rk2_innerstep(
                tnow, tnow+timestep,
                timestep/2,
                constants,
                shared_temp_equations,
                Nsystems, Nequations_per_system );

#ifdef ADAPTIVETIMESTEP
            // determine if any equation is above the absolute or relative tolerances
            if(fabs(y2 - y1) > ABSOLUTE_TOLERANCE || fabs((y2-y1)/(2*y2-y1+1e-12)) > RELATIVE_TOLERANCE){
                *shared_error_flag = 1;
                }
            __syncthreads();
#endif

            if (*shared_error_flag){
                // refine and start over
                timestep/=2;
                *shared_error_flag = 0;
            } // if shared_error_flag
            else{
                if (threadIdx.x == 0){
                    atomicAdd(nsteps,1);
                }
                //(*nsteps)++;
                // accept this step and update the shared array
                //  using local extrapolation (see NR e:17.2.3)
                shared_equations[threadIdx.x] = 2*y2-y1;
                tnow+=timestep;

#ifdef ADAPTIVETIMESTEP
                // let's get a little more optimistic
                timestep*=2;
#endif
            }// if shared_error_flag -> else

            __syncthreads();

        }// while tnow < tend

        // copy the y values back to global memory
        equations[tid]=shared_equations[threadIdx.x];
#ifdef LOUD
        if (threadIdx.x == 1 && blockIdx.x == 0){
            printf("nsteps taken: %d - tnow: %.2f\n",*nsteps,tnow);
        }
#endif
    } // if tid < nequations
} //integrate_rk2
