#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
int main(){

    void * rk2lib = dlopen("../../lib/rk2_gold.so", RTLD_LAZY);
    void * sielib = dlopen("../../lib/sie_gold.so", RTLD_LAZY);
    

    int (*p_goldIntegrateRK2)(float,float,int,float*,float*,int,int);
    p_goldIntegrateRK2  = dlsym(rk2lib,"goldIntegrateSystem");
    //p_goldIntegrateRK2  = dlsym(rk2lib,"_Z19cudaIntegrateSystemffiPfS_ii");

    int (*p_goldIntegrateSIE)(float,float,int,float*,float*,int,int);
    p_goldIntegrateSIE  = dlsym(sielib,"goldIntegrateSystem");

    int nsteps;
    nsteps = (*p_goldIntegrateSIE)(
        tnow, // the current time
        tend, // the time we integrating the system to
        n_integration_steps, // the initial timestep to attempt to integrate the system with
        constants, // the constants for each system
        equations, // a flattened array containing the y value for each equation in each system
        Nsystems, // the number of systems
        Neqn_p_sys);


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

    printf("SIEgold: %d nsteps\n",nsteps);


    tnow = 0;
    nsteps = (*p_goldIntegrateRK2)(
        tnow, // the current time
        tend, // the time we integrating the system to
        n_integration_steps, // the initial timestep to attempt to integrate the system with
        constants, // the constants for each system
        new_equations, // a flattened array containing the y value for each equation in each system
        Nsystems, // the number of systems
        Neqn_p_sys);

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
    printf("RK2gold: %d nsteps\n",nsteps);

    dlclose(rk2lib); 
    dlclose(sielib);
}
