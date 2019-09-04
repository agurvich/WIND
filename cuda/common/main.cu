#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <math.h> 

#include "wind_chimes.h"
#include "config.h"
#include "ode.h"

int main(int argc, const char *argv[]){ 
    int i; 
    struct globalVariables ChimesGlobalVars; 

    /* Define global variables. 
     * These are hard-coded for now - but 
     * it would be good to instead read 
     * these in from a parameter file. */ 
    strcpy(ChimesGlobalVars.MainDataTablePath, "/home/agurvich/src/CHIMES_repos/chimes-data/NewDataTables/chimes_main_data.hdf5"); 
    strcpy(ChimesGlobalVars.PhotoIonTablePath[0], "/home/agurvich/src/CHIMES_repos/chimes-data/NewDataTables/HM12_cross_sections/z0.000_cross_sections.hdf5"); 
    strcpy(ChimesGlobalVars.PhotoIonTablePath[1], "/home/agurvich/src/CHIMES_repos/chimes-data/NewDataTables/cross_sections_B87.hdf5"); 
    strcpy(ChimesGlobalVars.EqAbundanceTablePath, "/home/agurvich/src/CHIMES_repos/chimes-data/NewDataTables/EqAbundancesTables/DummyTable.hdf5"); 
    
    ChimesGlobalVars.cellSelfShieldingOn = 0; 
    ChimesGlobalVars.N_spectra = 2; 
    ChimesGlobalVars.StaticMolCooling = 1; 
    ChimesGlobalVars.T_mol = 1.0e3; 
    ChimesGlobalVars.InitIonState = 1; 
    ChimesGlobalVars.grain_temperature = 10.0; 
    ChimesGlobalVars.cmb_temperature = 2.725; 
    ChimesGlobalVars.relativeTolerance = 5.0e-3; 
    ChimesGlobalVars.absoluteTolerance = 5.0e-3; 
    ChimesGlobalVars.thermalAbsoluteTolerance = 1.0e-40; 
    ChimesGlobalVars.explicitTolerance = 0.01; 
    ChimesGlobalVars.scale_metal_tolerances = 0; 
    ChimesGlobalVars.n_threads = 1; 
    
    int Nsystems = 1;

    int use_metals = 0;
    float ABSOLUTE = 5e-3;
    float RELATIVE = 5e-3;

    int Nequations_per_system = 10;
    if (use_metals){
        Nequations_per_system = 157;
    }

    // set metal flags
    for (i = 0; i < 9; i++) 
        ChimesGlobalVars.element_included[i] = use_metals; 

    // initialize wind-chimes
    printf("Initializing WIND-CHIMES...\n");
    init_wind_chimes(&ChimesGlobalVars); 
    printf("... finished initializing WIND-CHIMES!\n");

    float tnow = 0;
    float tend = 10;
    int n_integration_steps = 1;
    WindFloat * constants;
    WindFloat * equations;

    constants = (WindFloat *) malloc(sizeof(WindFloat)*NUM_CONST*Nsystems);
    equations = (WindFloat *) malloc(sizeof(WindFloat)*Nequations_per_system*Nsystems);

    WindFloat base_constants[2] = {2,100};
    WindFloat base_equations[10] = {
        1.10032392, // ne
        0, // HI
        1, // HII
        0, // Hm
        0, // HeI
        9.92922857e-02, // HeII
        0, // HeIII
        0, // H2
        2.44161347e-04, // H2p
        0}; // H3p

    // tile our base system Nsystems many times
    for (int system_i=0; system_i<Nsystems;system_i++){
        for (int constant_i=0; constant_i<NUM_CONST; constant_i++){
            constants[system_i*NUM_CONST+constant_i]=base_constants[constant_i];
        }

        for (int equation_i=0; equation_i<Nequations_per_system; equation_i++){
            equations[system_i*Nequations_per_system+equation_i]=base_equations[equation_i];
        }
    }

    printf("Integrating %d systems of %d equations\n",Nsystems,Nequations_per_system);
    int nsteps = cudaIntegrateSystem(
        tnow, // the current time
        tend, // the time we integrating the system to
        n_integration_steps,//n_integration_steps,
        constants, // the constants for each system
        equations, // a flattened array containing the y value for each equation in each system
        Nsystems, // the number of systems
        Nequations_per_system, // the number of equations in each system
        ABSOLUTE, // the absolute tolerance
        RELATIVE); // the relative tolerance

    printf("%d steps final:\n",nsteps);
    for (int equation_i=0; equation_i<Nequations_per_system; equation_i++){
        printf("%.2e\t",equations[equation_i]);
    }
    printf("\n");
} 
