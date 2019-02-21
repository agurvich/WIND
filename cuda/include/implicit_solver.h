// SIE_solver (BDF1_solver)
void SIE_step(
    float *, // device pointer to the current timestep (across all systems, lame!!)
    float **,  // Nsystems x Neqn_p_sys*Neqn_p_sys 2d array with flattened jacobians
    float **, // Nsystems x Neqn_p_sys*Neqn_p_sys 2d array to store output (same as jacobians to overwrite)
    float **, // 1 x Neqn_p_sys*Neqn_p_sys array storing the identity (ideally in constant memory?)
    float **, // Nsystems x Neqn_p_sys 2d array to store derivatives
    float *, // Nsystems*Neqn_p_sys 1d array (flattened above)
    float *, // output state vector, iterative calls integrates
    int, // number of ODE systems
    int); // number of equations in each system


// BDF2_solver
void BDF2_step(
    float *, // device pointer to the current timestep (across all systems, lame!!)
    float **,  // Nsystems x Neqn_p_sys*Neqn_p_sys 2d array with flattened jacobians
    float **, // Nsystems x Neqn_p_sys*Neqn_p_sys 2d array to store output (same as jacobians to overwrite)
    float **, // 1 x Neqn_p_sys*Neqn_p_sys array storing the identity (ideally in constant memory?)
    float **, // Nsystems x Neqn_p_sys 2d array to store derivatives
    float *, // Nsystems*Neqn_p_sys 1d array (flattened above)
    float *, // state vector from previous timestep
    float *, // state vector from this timestep, where output goes
    float **, // matrix holding intermediate values used internally for the calculation
    float *, // flat array storing intermediate values used internally for the calculation
    int, // number of ODE systems
    int);// number of equations in each system
