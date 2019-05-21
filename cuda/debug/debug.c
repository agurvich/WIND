#include "inputKatz96_1_1_1.h"
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
int main(){

    void * rk2lib = dlopen("../lib/rk2.so", RTLD_LAZY);
    void * sielib = dlopen("../lib/sie.so", RTLD_LAZY);
    

    int (*p_cudaIntegrateRK2)(float,float,int,float*,float*,int,int,float,float);
    p_cudaIntegrateRK2  = dlsym(rk2lib,"_Z19cudaIntegrateSystemffiPfS_iiff");
    //p_cudaIntegrateRK2  = dlsym(rk2lib,"_Z16cudaIntegrateSIEffiPfS_ii");
    int (*p_cudaIntegrateSIE)(float,float,int,float*,float*,int,int,float,float);
    p_cudaIntegrateSIE  = dlsym(sielib,"_Z19cudaIntegrateSystemffiPfS_iiff");
    //p_cudaIntegrateSIE  = dlsym(sielib,"goldIntegrateSystem");


    int nsteps;
    nsteps = (*p_cudaIntegrateSIE)(
        tnow, // the current time
        tend, // the time we integrating the system to
        n_integration_steps, // the initial timestep to attempt to integrate the system with
        constants, // the constants for each system
        equations, // a flattened array containing the y value for each equation in each system
        Nsystems, // the number of systems
        Neqn_p_sys,
        5e-3,
        5e-3);

    printf("%.2f %.2f %.2f %.2f %.2f ",
        equations[0],
        equations[1],
        equations[2],
        equations[3],
        equations[4]);

    printf("%.2f %.2f %.2f %.2f %.2f\n",
        equations[5],
        equations[6],
        equations[7],
        equations[8],
        equations[9]);

    printf("SIE: %d nsteps\n",nsteps);


    tnow = 0;
    nsteps = (*p_cudaIntegrateRK2)(
        tnow, // the current time
        tend, // the time we integrating the system to
        n_integration_steps, // the initial timestep to attempt to integrate the system with
        constants, // the constants for each system
        new_equations, // a flattened array containing the y value for each equation in each system
        Nsystems, // the number of systems
        Neqn_p_sys,
        5e-3,
        5e-3);

    printf("%.2f %.2f %.2f %.2f %.2f ",
        new_equations[0],
        new_equations[1],
        new_equations[2],
        new_equations[3],
        new_equations[4]);

    printf("%.2f %.2f %.2f %.2f %.2f\n",
        new_equations[5],
        new_equations[6],
        new_equations[7],
        new_equations[8],
        new_equations[9]);
    printf("RK2: %d nsteps\n",nsteps);

    dlclose(rk2lib); 
    dlclose(sielib);
}
