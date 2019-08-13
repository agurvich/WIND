#include "inputKatz96_1_1_1.h"
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
int main(){

    void * rk2lib = dlopen("../lib/rk2.so", RTLD_LAZY);
    //void * rk2lib = dlopen("../lib/sie_gold.so", RTLD_LAZY);
    void * sielib = dlopen("../lib/sie.so", RTLD_LAZY);
    //void * sielib = dlopen("../lib/rk2_gold.so", RTLD_LAZY);
    

    int (*p_cudaIntegrateRK2)(float,float,int,float*,float*,int,int,float,float);
    p_cudaIntegrateRK2  = dlsym(rk2lib,"_Z19cudaIntegrateSystemffiPfS_iiff");
    //p_cudaIntegrateRK2  = dlsym(rk2lib,"goldIntegrateSystem");
    int (*p_cudaIntegrateSIE)(float,float,int,float*,float*,int,int,float,float);
    p_cudaIntegrateSIE  = dlsym(sielib,"_Z19cudaIntegrateSystemffiPfS_iiff");
    //p_cudaIntegrateSIE  = dlsym(sielib,"goldIntegrateSystem");

    tend=10;

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

    for (int i=0; i<Neqn_p_sys; i++){
        printf("%.2f ",
            equations[i]);
    }
    printf("\n");

    //printf("gold: %d nsteps\n",nsteps);
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

    for (int i=0; i<Neqn_p_sys; i++){
        printf("%.2f ",
            new_equations[i]);
    }
    printf("\n");

    //printf("gold: %d nsteps\n",nsteps);
    printf("RK2: %d nsteps\n",nsteps);

    dlclose(rk2lib); 
    dlclose(sielib);
}
