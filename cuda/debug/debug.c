#include "implicit_solver.h"
#include <stdio.h>
#include <dlfcn.h>

int main(){
    printf("hello world\n");

    float equations[20] = {
        0 , 1. , 0. , 0.09929229 , 0. ,
        0 , 1. , 0. , 0.09929229 , 0. ,
        0 , 1. , 0. , 0.09929229 , 0. ,
        0 , 1. , 0. , 0.09929229 , 0.};

    float constants[40] = {
        0 , 1.386e-03 , 4.1869916e-02 , 0 , 1.1655e-04 , 0 , 5.3550002e-07 , 2.5339721e-02 , 0 , 1.6747966e-01,
        0 , 1.386e-03 , 6.6359341e-02 , 0 , 1.1655e-04 , 0 , 5.3550002e-07 , 4.0160749e-02 , 0 , 2.6543736e-01,
        0 , 1.386e-03 , 3.0313825e-02 , 0 , 1.1655e-04 , 0 , 5.3550002e-07 , 1.8912189e-02 , 0 , 1.2125530e-01,
        0 , 1.386e-03 , 4.8044175e-02 , 0 , 1.1655e-04 , 0 , 5.3550002e-07 , 2.9973801e-02 , 0 , 1.9217670e-01};
 

    float tnow = 0;
    float tend = 10.0;
    int n_integration_steps = 1;
    int Nsystems = 4;
    int Neqn_p_sys = 5;



    void * sielib = dlopen("../lib/sie.so", RTLD_LAZY);
    
    int (*p_cudaIntegrateSIE)(float,float,int,float*,float*,int,int);
    p_cudaIntegrateSIE  = dlsym(sielib,"_Z16cudaIntegrateRK2ffiPfS_ii");

    printf("%.2f\n",equations[1]);
    printf("%.2e\n",constants[1]);


    int nsteps = (*p_cudaIntegrateSIE)(
        tnow, // the current time
        tend, // the time we integrating the system to
        n_integration_steps, // the initial timestep to attempt to integrate the system with
        constants, // the constants for each system
        equations, // a flattened array containing the y value for each equation in each system
        Nsystems, // the number of systems
        Neqn_p_sys);

    printf("%d nsteps\n",nsteps);

    dlclose(sielib); 
}
