const int THREAD_BLOCK_LIMIT = 1024;
__global__ void integrateSystem(
    float, // tnow
    float, // tend
    float, // timestep
    float *, // equations_flat
    float *, // constants_flat
    int, // Nsystems
    int, // Neqn_p_sys
    int *);  // nsteps
