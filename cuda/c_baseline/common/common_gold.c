#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "common_gold.h"

int integrateSystem(
    float tnow,
    float tend,
    float timestep,
    float * equations,
    float * constants,

    float * jacobians_flat, // NULL for rk2
    float * inverses_flat, // NULL for rk2

    int Neqn_p_sys){
    
    
    // allocate place to store full and half step
    float * y1 = (float *) malloc(sizeof(float)*Neqn_p_sys);
    float * y2 = (float *) malloc(sizeof(float)*Neqn_p_sys);
    float * dydt = (float *) malloc(sizeof(float)*Neqn_p_sys);

    // copy the values of the system to the temporary y1
    memcpy((void *)y1,(void *)equations,sizeof(float)*Neqn_p_sys);
    // copy the values of the system to the temporary y1
    memcpy((void *)y2,(void *)equations,sizeof(float)*Neqn_p_sys);

    // begin the error loop
    int nsteps =0;
    int unsolved = 0;
    int error_flag = 0;
    while (tnow < tend && unsolved<20){
        // make sure we don't overintegrate
        timestep = fmin(tend-tnow,timestep);

        // solve the full step 
        nsteps += take_step(
            tnow,tnow+timestep,
            1,
            y1,
            constants,
            dydt,
            jacobians_flat,
            inverses_flat,
            Neqn_p_sys);

        // solve the half step wrt y1
        nsteps+=take_step(
            tnow,tnow+timestep,
            2,
            y2,
            constants,
            dydt,
            jacobians_flat,
            inverses_flat,
            Neqn_p_sys);
#ifdef DEBUGBLOCK
            printf("%02d - gold - y1\t",nsteps);
            for (int eqn_i=0; eqn_i < Neqn_p_sys; eqn_i++){
                printf("%.6f\t",y1[eqn_i]);
            }
            printf("\n");
            printf("%02d - gold - y2\t",nsteps);
            for (int eqn_i=0; eqn_i < Neqn_p_sys; eqn_i++){
                printf("%.6f\t",y2[eqn_i]);
            }
            printf("\n");
#endif


#ifdef ADAPTIVE_TIMESTEP
        error_flag = checkError(y1,y2,Neqn_p_sys);
#endif
        if (error_flag && unsolved <10){
            // refine the timestep
            timestep/=2.0;
            // increment the number of attempts we made
            unsolved++;
            // copy the values of the system to the temporary y1
            memcpy((void *)y1,(void *)equations,sizeof(float)*Neqn_p_sys);
            // copy the values of the system to the temporary y2
            memcpy((void *)y2,(void *)equations,sizeof(float)*Neqn_p_sys);
        }
        else{
            // generically accept the solution, in general
            //  will want y2 but maybe a combination of y1 and y2 will
            //  cancel out error terms, e.g. rk2 

            acceptSolution(y1,y2,equations,Neqn_p_sys);

            tnow+=timestep;


            // get a little more ambitious
            timestep*=2.0;
            unsolved =0;
        }
    }

    return nsteps;
} 
