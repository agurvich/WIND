#include "ode.h"

__global__ void calculateDerivatives(
    float * d_derivatives_flat, 
    float * constants, 
    float * equations,
    int Neqn_p_sys,
    float time){
    // isolate this system 
    int eqn_offset = blockIdx.x*Neqn_p_sys;
    float * this_block_state = equations+eqn_offset;
    float * this_block_derivatives = d_derivatives_flat+eqn_offset;

    // constraint equation, ne = nH+ + nHe+ + 2*nHe++
    float ne = equations[eqn_offset+1]+equations[eqn_offset+3]+equations[eqn_offset+4]*2.0;

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
    this_block_derivatives[0] = constants[2]*ne*this_block_state[1]-(constants[0]*ne + constants[1])*this_block_state[0]; 

    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[1] = -this_block_derivatives[0];

    // He0 :(alpha_(He+)+alpha_(d)) ne nH+ - (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
    this_block_derivatives[2] = (constants[7]+constants[8])*ne*this_block_state[3] - (constants[3]*ne+constants[4])*this_block_state[2];

    // He+ : alpha_(He++) ne nHe++ + (Gamma_(e,He0)ne + Gamma_(gamma,He0)) nHe0
    // - (alpha_(He+)+alpha_(d)) ne nH+ - (Gamma_(e,He+)ne + Gamma_(gamma,He+)) nH+
    this_block_derivatives[3] = constants[9]*ne*this_block_state[4] + (constants[3]*ne+constants[4])*this_block_state[2] - 
        -(constants[7]+constants[8])*ne*this_block_state[3] - (constants[5]*ne+constants[6])*this_block_state[3];

    // He++ : -alpha_(He++) ne nHe++
    this_block_derivatives[4] = -constants[9]*ne*this_block_state[4]; 
}

__global__ void calculateJacobians(
    float **d_Jacobianss, 
    float * constants,
    float * equations,
    int Neqn_p_sys,
    float time){

    // isolate this system 
    int eqn_offset = blockIdx.x*Neqn_p_sys;
    float * this_block_state = equations+eqn_offset;
    float * this_block_jacobian = d_Jacobianss[blockIdx.x];

    // constraint equation, ne = nH+ + nHe+ + 2*nHe++
    float ne = this_block_state[1]+this_block_state[3]+this_block_state[4]*2.0;

    /* constants = [
        Gamma_(e,H0), Gamma_(gamma,H0), 
        alpha_(H+),
        Gamma_(e,He0), Gamma_(gamma,He0), 
        Gamma_(e,He+), Gamma_(gamma,He+),
            alpha_(He+), alpha_(d),
        alpha_(He++)
        ] 
    */

    // H0
    this_block_jacobian[0] = constants[0]*ne + constants[1]; // H0 : Gamma_(e,H0)ne + Gamma_(gamma,H0)
    this_block_jacobian[5] = -this_block_jacobian[0]; // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))

    //H+
    this_block_jacobian[1] = constants[2]*ne; // H0 : alpha_(H+)ne
    this_block_jacobian[6] = -this_block_jacobian[1]; // H+ -alpha_(H+)ne

    // He0
    this_block_jacobian[17] = constants[3]*ne+constants[4]; //He+ : Gamma_(e,He0)ne + Gamma_(gamma,He0)
    this_block_jacobian[12] = -this_block_jacobian[17]; //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))

    // He+
    this_block_jacobian[23] = constants[5]*ne+constants[6]; //He++ : Gamma_(e,He+)ne + Gamma_(gamma,He+)
    this_block_jacobian[13] = (constants[7]+constants[8])*ne; //He0 : (alpha_(He+)+alpha_(d))ne
    this_block_jacobian[18] = -this_block_jacobian[13] - 
        this_block_jacobian[23]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))

    // He++
    this_block_jacobian[19] = constants[9]*ne;//He+ : alpha_(He++)ne
    this_block_jacobian[24] = -this_block_jacobian[19];//He++ : -alpha_(He++)ne
}
