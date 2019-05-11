#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "common_gold.h"

int checkError(float * y1, float * y2, int Neqn_p_sys){
    double abs_error;
    double rel_error;
    int error_flag = 0;
    for (int eqn_i=0; eqn_i<Neqn_p_sys; eqn_i++){
        abs_error = fabs(y2[eqn_i]-y1[eqn_i]);
        if (abs_error >= ABSOLUTE_TOLERANCE){
#ifdef LOUD
            printf("%d absolute failed: %.2e\n",eqn_i,abs_error);
#endif
            error_flag = 1;
        }
        //if (fabs(y1[eqn_i]) > ABSOLUTE_TOLERANCE && 
            //fabs(y2[eqn_i]) > ABSOLUTE_TOLERANCE){
            //rel_error = abs_error/fmin(fabs(y1[eqn_i]),fabs(y2[eqn_i]));
        rel_error = fabs((y2[eqn_i] - y1[eqn_i])/(2*y2[eqn_i]-y1[eqn_i]+1e-12));
        if (rel_error >= RELATIVE_TOLERANCE){
#ifdef LOUD
                printf("%d relative failed: %.2e\n",eqn_i,rel_error);
#endif
                error_flag = 1;
        }// if rel_error >=RELATIVE_TOLERANCE
        //}// if fabs(y1) > ABS_TOL && fabs(y2) > ABS_TOL
    }// for eqn_i <Neqn_p_sys
    return error_flag;
}// int checkError

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
    while (tnow < tend && unsolved<10){
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
/*
            printf("tnow: %.4f timestep: %.4f nsteps: %d\n",tnow,timestep,nsteps);
*/
            tnow+=timestep;


            // get a little more ambitious
            timestep*=2.0;
            unsolved =0;
        }
    }

    return nsteps;
} 
