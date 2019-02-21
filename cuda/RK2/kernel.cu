#include <stdio.h>
#include <math.h>

#include "explicit_solver.h"

#define ABSOLUTE_TOLERANCE 1e-6
#define RELATIVE_TOLERANCE 1e-6

__device__ float calculate_dydt(
    float tnow,
    float * constants,
    float * equations){
    // constraint equation, ne = nH+ + nHe+ + 2*nHe++
    float ne = equations[1]+equations[3]+equations[4]*2.0;

    /* constants = [
        Gamma_(e,H0), Gamma_(gamma,H0), 
        alpha_(H+),
        Gamma_(e,He0), Gamma_(gamma,He0), 
        Gamma_(e,He+), Gamma_(gamma,He+),
        alpha_(He+),
        alpha_(d),
        alpha_(He++)
        ] 
    */

    if (threadIdx.x == 0){
        // H0 : alpha_(H+) ne nH+ - (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0
        return constants[2]*ne*equations[1]
            -(constants[0]*ne + constants[1])*equations[0]; 
    }
    else if (threadIdx.x == 1){
        // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
        return -constants[2]*ne*equations[1]
            +(constants[0]*ne + constants[1])*equations[0]; 
    }
    else if (threadIdx.x == 2){
        // He0 :(alpha_(He+)+alpha_(d)) ne nHe+ - (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
        return (constants[7]+constants[8])*ne*equations[3] 
            - (constants[3]*ne+constants[4])*equations[2];
    }
    else if (threadIdx.x == 3){
        // He+ : 
        //  alpha_(He++) ne nHe++ 
        //  + (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
        //  - (alpha_(He+)+alpha_(d)) ne nHe+ 
        //  - (Gamma_(e,He+)ne + Gamma_(gamma,He+)) nHe+
        return constants[9]*ne*equations[4] 
            + (constants[3]*ne+constants[4])*equations[2]  
            - (constants[7]+constants[8])*ne*equations[3] 
            - (constants[5]*ne+constants[6])*equations[3];
    }
    else if (threadIdx.x == 4){
        // He++ : -alpha_(He++) ne nHe++
        return -constants[9]*ne*equations[4];
    }
    else{
        return NULL;
    }
} // calculate_dydt

__device__ float rk2_innerstep(
    float tnow, // the current time
    float tstop, // the time we want to stop
    float h, // the timestep to take
    float * constants, // the constants for each system
    float * shared_temp_equations, // place to store temporary equations
    int Nsystems, // the number of systems
    int Nequations_per_system){ // the number of equations in each system

    float dydt = 0;

    while (tnow < tstop){
        // limit step size based on remaining time
        h = fmin(tstop - tnow, h);

        //calculate the derivative for this equation
        dydt = calculate_dydt(
            tnow,
            constants,
            shared_temp_equations);

        // update value of temporary equations
        shared_temp_equations[threadIdx.x] += h*dydt;
        tnow+=h;

    } // while(tnow < tstop)
    return shared_temp_equations[threadIdx.x];
}// rk2_innerstep

__global__ void integrate_rk2(
    float tnow, // the current time
    float tend, // the time we integrating the system to
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
    float h = (tend-tnow);

    // ensure thread within limit
    if (tid < Nsystems*Nequations_per_system ) {
        // copy the y values to shared memory
        shared_equations[threadIdx.x] = equations[tid];
        *shared_error_flag = 0;
        __syncthreads();

        //printf("%d thread %d block\n",threadIdx.x,blockIdx.x);
        while (tnow < tend){
            // make sure we don't overintegrate
            h = fmin(tend-tnow,h);

            // now reset the temporary equations
            shared_temp_equations[threadIdx.x] = shared_equations[threadIdx.x];
            __syncthreads();

            y1 = rk2_innerstep(
                tnow, tnow+h,
                h,
                constants,
                shared_temp_equations,
                Nsystems, Nequations_per_system );
            
            // now reset the temporary equations
            shared_temp_equations[threadIdx.x] = shared_equations[threadIdx.x];
            __syncthreads();

            y2 = rk2_innerstep(
                tnow, tnow+h,
                h/2,
                constants,
                shared_temp_equations,
                Nsystems, Nequations_per_system );

            // determine if any equation is above the absolute or relative tolerances
            if(fabs(y2 - y1) > ABSOLUTE_TOLERANCE || fabs((y2-y1)/(2*y2-y1+1e-12)) > RELATIVE_TOLERANCE){
                *shared_error_flag = 1;
                }
            __syncthreads();

            if (*shared_error_flag){
                // refine and start over
                h/=2;
                *shared_error_flag = 0;
            } // if shared_error_flag
            else{
                (*nsteps)++;
                // accept this step and update the shared array
                //  using local extrapolation (see NR e:17.2.3)
                shared_equations[threadIdx.x] = 2*y2-y1;
                tnow+=h;

                // let's get a little more optimistic
                h*=2;
            }// if shared_error_flag -> else

            __syncthreads();

        }// while tnow < tend

        // copy the y values back to global memory
        equations[tid]=shared_equations[threadIdx.x];
        if (threadIdx.x == 1 && blockIdx.x == 0){
            printf("nsteps taken: %d - tnow: %.2f\n",*nsteps,tnow);
        }
    } // if tid < nequations
} //integrate_rk2
