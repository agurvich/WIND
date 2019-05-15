#include "ode_gold.h"

void calculate_dydt(
    float tnow,
    float * equations,
    float * constants,
    float * dydt,
    int Neqn_p_sys){
/* ----- PREFIX FLAG FOR PYTHON FRONTEND ----- */

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
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    dydt[4] = (constants[5]*ne+constants[6])*equations[3]
        -constants[9]*ne*equations[4];
/* ----- SUFFIX FLAG FOR PYTHON FRONTEND ----- */
}

void calculate_jacobian(
    float * equations,
    float * constants,
    float * Jacobian){
/* ----- PREFIX FLAG FOR PYTHON FRONTEND ----- */
// constraint equation, ne = nH+ + nHe+ + 2*nHe++
    float ne = equations[1]+equations[3]+equations[4]*2.0;

    /* constants = [
        0-Gamma_(e,H0), 1-Gamma_(gamma,H0), 
        2-alpha_(H+),
        3-Gamma_(e,He0), 4-Gamma_(gamma,He0), 
        5-Gamma_(e,He+), 6-Gamma_(gamma,He+),
        7-alpha_(He+),
        8-alpha_(d),
        9-alpha_(He++)
        ] 
    */


   
    // H0
    Jacobian[0] = -(constants[0]*ne + constants[1]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    Jacobian[1] = -Jacobian[0]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
        
    //H+
    Jacobian[6] = -constants[2]*ne; // H+ -alpha_(H+)ne
    Jacobian[5] = -Jacobian[6]; // H0 : 2-alpha_(H+)ne
        
    // He0
    Jacobian[12] = -(constants[3]*ne+constants[4]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    Jacobian[13] = Jacobian[12]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
        
    // He+
    Jacobian[19] = constants[5]*ne+constants[6]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    Jacobian[17] = (constants[7]+constants[8])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    Jacobian[18] = -Jacobian[17] - 
        Jacobian[19]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
        
    // He++
    Jacobian[24] = -constants[9]*ne;//He++ : -alpha_(He++)ne
    Jacobian[23] = -Jacobian[24];//He+ : 9-alpha_(He++)ne
/* ----- SUFFIX FLAG FOR PYTHON FRONTEND ----- */
}
