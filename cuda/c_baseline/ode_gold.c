#include "ode_gold.h"

void calculate_dydt(
    float tnow,
    float * equations,
    float * constants,
    float * dydt,
    int Neqn_p_sys){

    // constraint equation, ne = nH+ + nHe+ + 2*nHe++
    float ne = equations[1]+equations[3]+equations[4]*2.0;

    /* constants = [
        Gamma_(e,H0), Gamma_(gamma,H0), 
        alpha_(H+),
        Gamma_(e,He0), Gamma_(gamma,He0), 
        Gamma_(e,He+), Gamma_(gamma,He+),
        alpha_(He+),
        alpha_(d),
        alpha_(He++)
        ] 
    */ 

    // H0 : alpha_(H+) ne nH+ - (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0
    dydt[0] = constants[2]*ne*equations[1]
            -(constants[0]*ne + constants[1])*equations[0]; 
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    dydt[1] = -constants[2]*ne*equations[1]
            +(constants[0]*ne + constants[1])*equations[0]; 
    // He0 :(alpha_(He+)+alpha_(d)) ne nHe+ - (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
    dydt[2] = (constants[7]+constants[8])*ne*equations[3] 
        - (constants[3]*ne+constants[4])*equations[2];
        // He+ : 
        //  alpha_(He++) ne nHe++ 
        //  + (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
        //  - (alpha_(He+)+alpha_(d)) ne nHe+ 
        //  - (Gamma_(e,He+)ne + Gamma_(gamma,He+)) nHe+
    dydt[3] = constants[9]*ne*equations[4] 
        + (constants[3]*ne+constants[4])*equations[2]  
        - (constants[7]+constants[8])*ne*equations[3] 
        - (constants[5]*ne+constants[6])*equations[3];
    dydt[4] = -constants[9]*ne*equations[4];
}


void calculate_jacobian(){

}
