// define top-level integration kernel
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
 
// define the device kernels 
/*
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
*/

__device__ WindFloat evaluate_RHS_function(
    float , //tnow,
    void * , //RHS_input,
    WindFloat *, // constants,
    WindFloat *, // shared_equations,
    WindFloat *, // shared_dydts,
    WindFloat *, // Jacobians,
    int); // Nequations_per_system)

extern void * RHS_input;
