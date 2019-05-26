#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
int main(){

    void * rk2lib = dlopen("../../lib/rk2_gold.so", RTLD_LAZY);
    void * sielib = dlopen("../../lib/sie_gold.so", RTLD_LAZY);
    

    int (*p_goldIntegrateRK2)(float,float,int,float*,float*,int,int,float,float);
    p_goldIntegrateRK2  = dlsym(rk2lib,"goldIntegrateSystem");
    //p_goldIntegrateRK2  = dlsym(rk2lib,"_Z19cudaIntegrateSystemffiPfS_ii");

    int (*p_goldIntegrateSIE)(float,float,int,float*,float*,int,int,float,float);
    p_goldIntegrateSIE  = dlsym(sielib,"goldIntegrateSystem");

    int nsteps;
    nsteps = (*p_goldIntegrateSIE)(
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

    printf("SIEgold: %d nsteps\n",nsteps);


    tnow = 0;
    nsteps = (*p_goldIntegrateRK2)(
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
            equations[i]);
    }
    printf("\n");

    printf("RK2gold: %d nsteps\n",nsteps);

    dlclose(rk2lib); 
    dlclose(sielib);
}
