/* ------ <solver>_gold.c ------ */
void acceptSolution(
    float *,// y1,
    float *,// y2,
    float *, // equations,
    int);// Neqn_p_sys);

int take_step(
    float,// tnow,
    float,// tend,
    int,// n_integration_steps,
    float *,// equations,
    float *,// constants,
    float *,// dydt,

    float *,// jacobians_flat
    float *,// inverses_flat

    int );//Neqn_p_sys);


/* ------ common_gold.c ------ */
int checkError(
    float *,// y1,
    float *,// y2,
    int,// Neqn_p_sys);
    float, //absolute
    float); //relative

int integrateSystem(
    float ,//tnow,
    float ,//tend,
    float ,//timestep
    float *,// equations,
    float *,// constants,

    float *,// jacobians_flat, // NULL for rk2
    float *,// inverses_flat, // NULL for rk2

    int, //Neqn_p_sys);
    float, // absolute
    float); // relative
