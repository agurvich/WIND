__device__ float calculate_dydt(
    float ,//tnow,
    float *, //constants,
    float * );//equations);


__device__ void calculate_jacobian(
    float ,//tnow,
    float *,// constants,
    float *,// shared_temp_equations,
    float *,// Jacobian)
    int ); // Neqn_p_sys

__global__ void read_texture(void *);

extern void * RHS_input;

#define NUM_CONST 2
