#include <stdio.h>
#include <string.h>
#include <math.h>

#include "common_gold.h"
#include "ode_gold.h"

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


void acceptSolution(float * y1, float * y2, float * equations,int Neqn_p_sys){
    for (int eqn_i=0; eqn_i<Neqn_p_sys; eqn_i++){
        // RK2 definition of how to accept the solution
        equations[eqn_i] = 2*y2[eqn_i] - y1[eqn_i];
    }
    // copy the value of equations into y1 & y2
    memcpy((void *)y1,(void *)equations,sizeof(float)*Neqn_p_sys);
    memcpy((void *)y2,(void *)equations,sizeof(float)*Neqn_p_sys);
}

int take_step(
    float tnow,
    float tend,
    int n_integration_steps,
    float * equations,
    float * constants,
    float * dydt,

    float * jacobians_flat, // NULL for rk2
    float * inverses_flat, // NULL for rk2

    int Neqn_p_sys){

    // calculate timestep
    float timestep = (tend-tnow)/n_integration_steps;

    for (int nsteps=0; nsteps<n_integration_steps; nsteps++){
        // fill the derivative vector
        calculate_dydt(tnow,equations,constants,dydt,Neqn_p_sys);

        // take the trial step
        for (int eqn_i=0; eqn_i<Neqn_p_sys; eqn_i++){
            equations[eqn_i]+=timestep*dydt[eqn_i];
        }

        tnow+=timestep;
    }
    return n_integration_steps;
}

int goldIntegrateSystem(
    float tnow, // the current time
    float tend, // the time we integrating the system to
    int n_integration_steps,
    float * constantss_flat, // the constants for each system
    float * equationss_flat, // a flattened array containing the y value for each equation in each system
    int Nsystems, // the number of systems
    int Neqn_p_sys){ // the number of equations in each system

#ifdef LOUD
    printf("RK2 Received %d systems, %d equations per system\n",Nsystems,Neqn_p_sys);
#endif

    int nloops=0;
    for (int system_i=0; system_i < Nsystems; system_i++){
        nloops+=integrateSystem(
            tnow,tend,
            (tend-tnow)/n_integration_steps, 
            equationss_flat + Neqn_p_sys*system_i,
            constantss_flat + NUMCONST*system_i,
            NULL,
            NULL,
            Neqn_p_sys);
    }
    
    // return how many steps were taken
    return nloops;
} // goldIntegrateSystem
