#include <stdio.h>
#include <math.h>

#include "explicit_solver.h"
#include "device.h"
#include "ode.h"

__device__ void checkError(float y1, float y2, int * shared_error_flag){
    // determine if any equation is above the absolute or relative tolerances
    float abs_error = fabs(y2 - y1);
    if(abs_error > ABSOLUTE_TOLERANCE){
        *shared_error_flag = 1;
#ifdef LOUD
        printf("%d absolute failed: %.2e\n",threadIdx.x,abs_error);
#endif
    }
    float rel_error = fabs((y2-y1)/(2*y2-y1+1e-12));
    if(rel_error > RELATIVE_TOLERANCE){
        *shared_error_flag = 1;
#ifdef LOUD
        printf("%d relative failed: %.2e\n",threadIdx.x,rel_error);
#endif
        }
    __syncthreads();
}

__device__ float rk2_innerstep(
    float tnow, // the current time
    float tstop, // the time we want to stop
    int n_integration_steps, // the timestep to take
    float * constants, // the constants for each system
    float * shared_temp_equations, // place to store temporary equations
    int Nsystems, // the number of systems
    int Nequations_per_system){ // the number of equations in each system

    float dydt = 0;

    float timestep = (tstop-tnow)/n_integration_steps;
    for (int nsteps=0; nsteps<n_integration_steps; nsteps++){
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

__global__ void integrateSystem(
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

    // offset pointer to constants in global memory
    constants += NUM_CONST*blockIdx.x;

    float y1,y2;

    int this_nsteps = 0;
    // ensure thread within limit
    if (tid < Nsystems*Nequations_per_system ) {
        // copy the y values to shared memory
        shared_equations[threadIdx.x] = equations[tid];
        *shared_error_flag = 0;
        __syncthreads();

        //printf("%d thread %d block\n",threadIdx.x,blockIdx.x);
        while (tnow < tend){
            this_nsteps+=3;
            // make sure we don't overintegrate
            timestep = fmin(tend-tnow,timestep);

            // now reset the temporary equations
            shared_temp_equations[threadIdx.x] = shared_equations[threadIdx.x];
            __syncthreads();

            y1 = rk2_innerstep(
                tnow, tnow+timestep,
                1,
                constants,
                shared_temp_equations,
                Nsystems, Nequations_per_system );
/*
            if (threadIdx.x==0 && blockIdx.x==1){
                printf("%02d - y1: ",this_nsteps);
                printf("%.6f\t",shared_temp_equations[0]);
                printf("%.6f\t",shared_temp_equations[1]);
                printf("%.6f\t",shared_temp_equations[2]);
                printf("%.6f\t",shared_temp_equations[3]);
                printf("%.6f\t",shared_temp_equations[4]);
                printf("\n");
            }
*/
            
            // now reset the temporary equations
            shared_temp_equations[threadIdx.x] = shared_equations[threadIdx.x];
            __syncthreads();

            y2 = rk2_innerstep(
                tnow, tnow+timestep,
                2,
                constants,
                shared_temp_equations,
                Nsystems, Nequations_per_system );
/*
            if (threadIdx.x==0 && blockIdx.x==1){
                printf("%02d - y2: ",this_nsteps);
                printf("%.6f\t",shared_temp_equations[0]);
                printf("%.6f\t",shared_temp_equations[1]);
                printf("%.6f\t",shared_temp_equations[2]);
                printf("%.6f\t",shared_temp_equations[3]);
                printf("%.6f\t",shared_temp_equations[4]);
                printf("\n");
                printf("\n");
            }
*/

#ifdef ADAPTIVE_TIMESTEP
            checkError(y1,y2,shared_error_flag); 
#endif

            if (*shared_error_flag){
                // refine and start over
                timestep/=2;
                *shared_error_flag = 0;
            } // if shared_error_flag
            else{
                //(*nsteps)++;
                // accept this step and update the shared array
                //  using local extrapolation (see NR e:17.2.3)
                shared_equations[threadIdx.x] = 2*y2-y1;

/*
                if (threadIdx.x==0 && blockIdx.x==3){
                    printf("tnow: %.4f timestep: %.4f nsteps: %d bid: %d\n",tnow,timestep,this_nsteps,blockIdx.x);
                }
*/

                tnow+=timestep;

#ifdef ADAPTIVE_TIMESTEP
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
            printf("nsteps taken: %d - tnow: %.2f\n",this_nsteps,tnow);
        }
#endif

        if (threadIdx.x == 0){
            atomicAdd(nsteps,this_nsteps);
        }
    } // if tid < nequations
} //integrateSystem
