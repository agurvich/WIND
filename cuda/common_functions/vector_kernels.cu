#include "vector_kernels.h"
#include <math.h>
#include <stdio.h>

__device__ int get_vector_tid(){
    // assumes gridDim.y = Nsystems and 
    //  and    gridDim.x = 1+Neqn_p_sys/1024
    //  and    blockDim.x = min(Neqn_p_sys,1024)

    //        i_system    ~neqn_p_sys           ~this eqn
    int tid =  
                blockIdx.z*(blockDim.x*gridDim.x*gridDim.y) +
                blockIdx.y*(blockDim.x*gridDim.x) + 
                blockIdx.x*blockDim.x +
                threadIdx.x;
    return tid;
}
__global__ void overwriteVector(float * v1, float * v2, int Nsystems, int Neqn_p_sys){
    // copies the contents of v1 into v2
    int tid = get_vector_tid();
    if (tid<(Neqn_p_sys*Nsystems)){
        v2[tid] = v1[tid];
    }
}

__global__ void scaleVector(float * vector, float scale, int Nsystems, int Neqn_p_sys){
    int tid = get_vector_tid();
    if (tid<(Nsystems*Neqn_p_sys)){
        vector[tid]*=scale;
    }
}

__global__ void addVectors(
    float alpha, float * v1,
    float beta, float * v2,
    float * v3,
    int Nsystems, int Neqn_p_sys){
    // outputs the result in v3
    int tid = get_vector_tid();
    if (tid<(Neqn_p_sys*Nsystems)){
        v3[tid] = alpha * v1[tid] + beta * v2[tid];
    }
}

__global__ void checkError(
    float * v1, float * v2,
    int * bool_flag,
    int Nsystems, int Neqn_p_sys){
    // replace the values of v1 with the error
    int tid = get_vector_tid();
    if (tid < Nsystems*Neqn_p_sys){
        float abs_error = fabs(v1[tid]-v2[tid]);

        if (abs_error > ABSOLUTE_TOLERANCE){
#ifdef LOUD
            printf("ABSOLUTE %d %.2e v1 %.2e v2 \n",tid,v1[tid],v2[tid]);
#endif
            *bool_flag = 1;
        }
        if (fabs((v1[tid]-v2[tid])/(v2[tid]+1e-12)) > RELATIVE_TOLERANCE && 
            v1[tid] > ABSOLUTE_TOLERANCE &&
            v2[tid] > ABSOLUTE_TOLERANCE){
#ifdef LOUD
            printf("RELATIVE %d %.2e\n",tid,v1[tid]/v2[tid]);
#endif
            *bool_flag = 1;
        }
    }
}

__global__ void addArrayToBatchArrays(
    float ** single_arr,
    float ** batch_arrs,
    float alpha, float beta,
    float p_beta,
    int Nsystems,
    int Neqn_p_sys){
    // assumes that gridDim.y = Nsystems, and blockDim.x = Neqn_p_sys
    int bid = blockIdx.z*gridDim.y + blockIdx.y;
    int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if (tid < (Neqn_p_sys*Neqn_p_sys) && bid < Nsystems){
        batch_arrs[bid][tid]=alpha*single_arr[0][tid]+ beta*(p_beta)*batch_arrs[bid][tid];
    }
}

__global__ void updateTimestep(
    float * timestep,
    float * derivatives_flat,
    float * scale_factor,
    int * max_index){
    // changes the value of the pointer in global memory on the device without copying back the derivatives
    //float ABSOLUTE_TOLERANCE = 1e-4;
    // -1 because cublas is 1 index. whyyyy
    /*
    if ( derivatives_flat[*max_index-1] < 1000*ABSOLUTE_TOLERANCE){
        *timestep = 1000*ABSOLUTE_TOLERANCE;
    }
    else{
        *timestep = ABSOLUTE_TOLERANCE/derivatives_flat[*max_index-1];
    }
    */
    *timestep = 1 * (*scale_factor);

    // TODO fixed timestep because why not?
    //*timestep = 0.25;
}
