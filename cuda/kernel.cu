#include <stdio.h>
#include "explicit_solver.h"

#define ABSOLUTE_TOLERANCE 1e-2
#define RELATIVE_TOLERANCE 1e-2

__global__ void hello(int * a, int * b){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    printf("tid %d\n",tid);
    a[tid]=a[tid]+b[tid];
} // hello

__device__ float calculate_dydt(
    float tnow,
    float * constants,
    float * equations){
    // assumes that constant and equation are pointers that start 
    //  at the beginning of the block's values.
    int tid = blockIdx.x*blockDim.x+threadIdx.x;

    return constants[tid]*tnow*tnow; //equations[threadIdx.x]*equations[(threadIdx.x+1)%blockDim.x];
} // calculate_dydyt

__device__ float euler_innerstep(
    float tnow, // the current time
    float tstop, // the time we want to stop
    float h, // the timestep to take
    float * constants, // the constants for each system
    float * shared_temp_equations, // place to store temporary equations
    int Nsystems, // the number of systems
    int Nequations_per_system){ // the number of equations in each system

    float dydt = 0;

    //dydt = (float *) malloc(blockDim.x*sizeof(float));
    while (tnow < tstop){
        // limit step size based on remaining time
        h = fmin(tstop - tnow, h);

        //calculate the derivative for this equation
        dydt = calculate_dydt(
            tnow,
            constants,
            shared_temp_equations);

            if (threadIdx.x == 1 && blockIdx.x == 0){
                printf("dydt(%.2f) - t(%.2f) - h(%.2f) - dydt-inner-step-%d-%d\n",
                    dydt,
                    tnow,
                    h,
                    blockIdx.x,threadIdx.x);
            }

        // update value of temporary equations
        shared_temp_equations[threadIdx.x] += h*dydt;
        tnow+=h;

    } // while(tnow < tstop)
    return shared_temp_equations[threadIdx.x];
}// euler_innerstep

__global__ void integrate_euler(
    float tnow, // the current time
    float tend, // the time we integrating the system to
    float * constants, // the constants for each system
    float * equations, // a flattened array containing the y value for each equation in each system
    int Nsystems, // the number of systems
    int Nequations_per_system){ // the number of equations in each system

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
    int nloops = 0;

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

            y1 = euler_innerstep(
                tnow, tnow+h,
                h,
                constants,
                shared_temp_equations,
                Nsystems, Nequations_per_system );
            
            // now reset the temporary equations
            shared_temp_equations[threadIdx.x] = shared_equations[threadIdx.x];
            __syncthreads();
            /*
            if (threadIdx.x == 1 && blockIdx.x == 0){
                printf("y1 %.2f\n",y1);
            }
            */

            y2 = euler_innerstep(
                tnow, tnow+h,
                h/2,
                constants,
                shared_temp_equations,
                Nsystems, Nequations_per_system );

            /*
            if (threadIdx.x == 1 && blockIdx.x == 0){
                printf("y2 %.2f\n",y2);
            }
            */

            *shared_error_flag = y2 - y1 > ABSOLUTE_TOLERANCE || (y2-y1)/(2*y2-y1) > RELATIVE_TOLERANCE;
            __syncthreads();

            if (threadIdx.x == 1 && blockIdx.x == 0){
                printf("%.3f-%.3f - error(%d) - %d thread %d block\n",
                    y1,
                    y2,
                    *shared_error_flag,
                    threadIdx.x,
                    blockIdx.x);
            }

            //tnow+=h;
            nloops++;
            if (*shared_error_flag){
                if (threadIdx.x == 1 && blockIdx.x == 0){
                    printf("bad - %d\n",*shared_error_flag);
                }
                // refine and start over
                h/=2;
            } // if shared_error_flag
            else{
                if (threadIdx.x == 1 && blockIdx.x == 0){
                    printf("good\n");
                }
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
            printf("nloops: %d\n",nloops);
        }
    } // if tid < nequations
} //integrate_euler
