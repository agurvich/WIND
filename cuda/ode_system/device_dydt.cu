#include "config.h"
#include "ode.h"


__device__ WindFloat calculate_dydt(
    float tnow,
    WindFloat * constants,
    WindFloat * equations){
/* ----- PREFIX FLAG FOR PYTHON FRONTEND ----- */
    // constraint equation, ne = nH+ + nHe+ + 2*nHe++
    WindFloat ne = equations[1]+equations[3]+equations[4]*2.0;

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


    if (threadIdx.x == 0){
        // H0 : alpha_(H+) ne nH+ - (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0
        return constants[2]*ne*equations[1]
            -(constants[0]*ne + constants[1])*equations[0]; 
    }
    else if (threadIdx.x == 1){
        // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
        return -constants[2]*ne*equations[1]
            +(constants[0]*ne + constants[1])*equations[0]; 
    }
    else if (threadIdx.x == 2){
        // He0 :(alpha_(He+)+alpha_(d)) ne nHe+ - (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
        return (constants[7]+constants[8])*ne*equations[3] 
            - (constants[3]*ne+constants[4])*equations[2];
    }
    else if (threadIdx.x == 3){
        // He+ : 
        //  alpha_(He++) ne nHe++ 
        //  + (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
        //  - (alpha_(He+)+alpha_(d)) ne nHe+ 
        //  - (Gamma_(e,He+)ne + Gamma_(gamma,He+)) nHe+
        return constants[9]*ne*equations[4] 
            + (constants[3]*ne+constants[4])*equations[2]  
            - (constants[7]+constants[8])*ne*equations[3] 
            - (constants[5]*ne+constants[6])*equations[3];
    }
    else if (threadIdx.x == 4){
        // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
        //  - 9-alpha_(He++) ne nHe++
        return (constants[5]*ne+constants[6])*equations[3]
        -constants[9]*ne*equations[4];
    }
/* ----- SUFFIX FLAG FOR PYTHON FRONTEND ----- */
    
   else{
        return NULL;
    } 
} // calculate_dydt

__device__ void calculate_jacobian(
    float tnow,
    WindFloat * constants,
    WindFloat * equations,
    WindFloat * Jacobian,
    int Neqn_p_sys){
    if (threadIdx.x == 0){
/* ----- PREFIX FLAG FOR PYTHON FRONTEND ----- */

    // constraint equation, ne = nH+ + nHe+ + 2*nHe++
    WindFloat ne = equations[1]+equations[3]+equations[4]*2.0;

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

    // NOTE could make this faster if we could do it in paralell 


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

    __syncthreads();
} //calculate_jacobian 


__device__ WindFloat evaluate_RHS_function(
    float tnow,
    void * RHS_input,
    WindFloat * constants,
    WindFloat * shared_equations,
    WindFloat * shared_dydts,
    WindFloat * Jacobians,
    int Nequations_per_system){
    //calculate the derivative for this equation
    WindFloat dydt = calculate_dydt(
        tnow,
        constants,
        shared_equations);

#ifdef SIE

    // calculate the jacobian for the whole system
    calculate_jacobian(
        tnow,
        constants,
        shared_equations,
        Jacobians,
        Nequations_per_system);
#endif
    return dydt;
}
