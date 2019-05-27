#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "common_gold.h"
#include "ode_gold.h"
#include "linear_algebra.h"

void acceptSolution(float * y1, float * y2, float * equations,int Neqn_p_sys){
    // take the solution to be y2
    memcpy((void *)equations,(void *)y2,sizeof(float)*Neqn_p_sys);
    // copy the value of equations into y1
    memcpy((void *)y1,(void *)equations,sizeof(float)*Neqn_p_sys);
}

void scaleAndInvertJacobians(
    float * Jacobian,
    float * inverse,
    float timestep,
    int Neqn_p_sys){

     
    // turn J into 1-hJ
    for (int ele_i=0; ele_i<Neqn_p_sys*Neqn_p_sys; ele_i++){
        Jacobian[ele_i]=-timestep*Jacobian[ele_i];
    }
    int diag_index;
    for (int eqn_i=0; eqn_i<Neqn_p_sys; eqn_i++){
        diag_index = eqn_i*(Neqn_p_sys+1);
        Jacobian[diag_index]+=1;
    }
    
    // invert Jacobian into inverse
    gjeInvertMatrix(Jacobian,inverse,Neqn_p_sys);
}

int take_step(
    float tnow,
    float tend,
    int n_integration_steps,
    float * equations,
    float * constants,
    float * dydt,

    float * jacobians_flat, 
    float * inverses_flat,

    int Neqn_p_sys){

    // calculate timestep
    float timestep = (tend-tnow)/n_integration_steps;

    for (int nsteps=0; nsteps<n_integration_steps; nsteps++){
        // fill the derivative vector and Jacobian matrix
        calculate_dydt(tnow,equations,constants,dydt,Neqn_p_sys);
        
        // zero out the jacobian 
        memset(jacobians_flat,0,sizeof(float)*Neqn_p_sys*Neqn_p_sys);
        calculate_jacobian(equations,constants,jacobians_flat);
        // invert 1-hj into inverses
        scaleAndInvertJacobians(
            jacobians_flat,
            inverses_flat,
            timestep,
            Neqn_p_sys);

        // multiply (1-hJ)^-1 with f
        matrixVectorMult(inverses_flat,dydt,Neqn_p_sys);
        
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
    int Neqn_p_sys, // the number of equations in each system
    float ABSOLUTE,
    float RELATIVE){ 

#ifdef LOUD
    printf("SIEgold Received %d systems, %d equations per system\n",Nsystems,Neqn_p_sys);
#endif

    int nloops=0;
    float * Jacobian;
    float * inverse;
    Jacobian = (float *) malloc(sizeof(float)*Neqn_p_sys*Neqn_p_sys);
    inverse = (float *) malloc(sizeof(float)*Neqn_p_sys*Neqn_p_sys);

    for (int system_i=0; system_i < Nsystems; system_i++){
        nloops+=integrateSystem(
            tnow,tend,
            (tend-tnow)/n_integration_steps, 
            equationss_flat + Neqn_p_sys*system_i,
            constantss_flat + NUM_CONST*system_i,
            Jacobian,
            inverse,
            Neqn_p_sys,
            ABSOLUTE,
            RELATIVE);
    }
    
    free(Jacobian);
    free(inverse);
    // return how many steps were taken
    return nloops;
} // goldIntegrateSystem
