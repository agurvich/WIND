#include "implicit_solver.h"
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

int main(){
    printf("hello world\n");

    /*
    float equations[40] = {
        0 , 1. , 0. , 0.09929229 , 0. ,0 , 1. , 0. , 0.09929229 , 0. ,
        0 , 1. , 0. , 0.09929229 , 0. ,0 , 1. , 0. , 0.09929229 , 0. ,
        0 , 1. , 0. , 0.09929229 , 0. ,0 , 1. , 0. , 0.09929229 , 0. ,
        0 , 1. , 0. , 0.09929229 , 0. ,0 , 1. , 0. , 0.09929229 , 0.};

    float constants[40] = {
        0 , 1.386e-03 , 4.1869916e-02 , 0 , 1.1655e-04 , 0 , 5.3550002e-07 , 2.5339721e-02 , 0 , 1.6747966e-01,
        0 , 1.386e-03 , 6.6359341e-02 , 0 , 1.1655e-04 , 0 , 5.3550002e-07 , 4.0160749e-02 , 0 , 2.6543736e-01,
        0 , 1.386e-03 , 3.0313825e-02 , 0 , 1.1655e-04 , 0 , 5.3550002e-07 , 1.8912189e-02 , 0 , 1.2125530e-01,
        0 , 1.386e-03 , 4.8044175e-02 , 0 , 1.1655e-04 , 0 , 5.3550002e-07 , 2.9973801e-02 , 0 , 1.9217670e-01};
 

    float tnow = 0;
    float tend = 10.0;
    int n_integration_steps = 4;
    int Nsystems = 4;
    int Neqn_p_sys = 10;
    */

    float tnow = 0;
    float tend = 5;
    int n_integration_steps = 1;

    float equations[12] = {1.000e+00,0,1.000e+00,0,1.000e+00,0,1.000e+00,0,1.000e+00,0,1.000e+00,0};
    float new_equations[12] = {1.000e+00,0,1.000e+00,0,1.000e+00,0,1.000e+00,0,1.000e+00,0,1.000e+00,0};

    float constants[4] = {9.980e+02,1.998e+03,-9.990e+02,-1.999e+03};
    int Nsystems = 1;
    int Neqn_p_sys = 12;

    void * sielib = dlopen("../lib/sie.so", RTLD_LAZY);
    void * sie2lib = dlopen("../lib/sie2.so", RTLD_LAZY);
    
    int (*p_cudaIntegrateSIE)(float,float,int,float*,float*,int,int);
    p_cudaIntegrateSIE  = dlsym(sielib,"_Z16cudaIntegrateSIEffiPfS_ii");

    int (*p_cudaIntegrateSIM)(float,float,int,float*,float*,int,int);
    p_cudaIntegrateSIM  = dlsym(sie2lib,"_Z16cudaIntegrateSIEffiPfS_ii");

    int nsteps;
    nsteps = (*p_cudaIntegrateSIE)(
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

    printf("SIE: %d nsteps\n",nsteps);

    tnow = 0;
    nsteps = (*p_cudaIntegrateSIM)(
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
    printf("SIM: %d nsteps\n",nsteps);

    dlclose(sielib); 
    dlclose(sie2lib);
}
