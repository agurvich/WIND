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
;
