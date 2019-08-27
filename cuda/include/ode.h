// define original integration kernel
#ifdef SIE
__global__ void integrateSystem(
    float, // tnow
    float, // tend
    float, // timestep
    WindFloat *, // equations_flat
    WindFloat *, // constants_flat
    WindFloat *, // Jacobians_flat 
    WindFloat *, // Inverses_flat 
    int, // Nsystems
    int, // Neqn_p_sys
    int *,// nsteps
    float, // absolute tolerance
    float); // absolute_tolerance
#else
__global__ void integrateSystem(
    float, // tnow
    float, // tend
    float, // timestep
    WindFloat *, // equations_flat
    WindFloat *, // constants_flat
    int, // Nsystems
    int, // Neqn_p_sys
    int *,  // nsteps
    float,// absolute tolerance
    float); // relative tolerance
#endif 


// define the device kernels 
__device__ WindFloat calculate_dydt(
    float ,//tnow,
    WindFloat *, //constants,
    WindFloat * );//equations);

__device__ void calculate_jacobian(
    float ,//tnow,
    WindFloat *,// constants,
    WindFloat *,// shared_temp_equations,
    WindFloat *,// Jacobian)
    int ); // Neqn_p_sys

__global__ void read_texture(void *);

extern void * RHS_input;
