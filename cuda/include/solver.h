const int THREAD_BLOCK_LIMIT = 1024;
#ifdef SIE
__global__ void integrateSystem(
    float, // tnow
    float, // tend
    float, // timestep
    float *, // equations_flat
    float *, // constants_flat
    float *, // Jacobians_flat 
    float *, // Inverses_flat 
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
    float *, // equations_flat
    float *, // constants_flat
    //float *, // Jacobians_flat 
    //float *, // Inverses_flat 
    int, // Nsystems
    int, // Neqn_p_sys
    int *,  // nsteps
    float,// absolute tolerance
    float); // relative tolerance
#endif 
