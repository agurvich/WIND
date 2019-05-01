
import sys
import numpy as np
from preprocess_words import derivative_prefix,derivative_suffix
from preprocess_words import jacobian_prefix,jacobian_suffix
from preprocess_words import file_prefix,file_suffix



def reindex(index,Ntile,Ndim,this_tile):
    nrow = index // Ndim
    return index + (Ntile-1)*Ndim*nrow + (Ndim*Ndim*Ntile+Ndim)*this_tile


def make_jacobian_block(this_tile,Ntile):
    ridx = lambda x: reindex(x,Ntile,5,this_tile)
    jacobian_good_stuff = ("""   
    // H0
    this_block_jacobian[%d] = -(constants[%d]*ne + constants[%d]); // H+ : -(Gamma_(e,H0)ne + Gamma_(gamma,H0))
    this_block_jacobian[%d] = -this_block_jacobian[%d]; // H0 : 0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0)
    """%(ridx(0),0,1,ridx(1),ridx(0)) + 
    """
    //H+
    this_block_jacobian[%d] = -constants[%d]*ne; // H+ -alpha_(H+)ne
    this_block_jacobian[%d] = -this_block_jacobian[%d]; // H0 : 2-alpha_(H+)ne
    """%(ridx(6),2,ridx(5),ridx(6)) + 
    """
    // He0
    this_block_jacobian[%d] = -(constants[%d]*ne+constants[%d]); //He0 : -(Gamma_(e,He0)ne + Gamma_(gamma,He0))
    this_block_jacobian[%d] = this_block_jacobian[%d]; //He+ : 3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)
    """%(ridx(12),3,4,ridx(13),ridx(12))  + 
    """
    // He+
    this_block_jacobian[%d] = constants[%d]*ne+constants[%d]; //He++ : 5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)
    this_block_jacobian[%d] = (constants[%d]+constants[%d])*ne; //He0 : (7-alpha_(He+)+8-alpha_(d))ne
    this_block_jacobian[%d] = -this_block_jacobian[%d] - 
        this_block_jacobian[%d]; //He+ : -((alpha_(He+)+alpha_(d)+Gamma_(e,He+))ne+Gamma_(gamma,He+))
    """%(ridx(19),5,6,ridx(17),7,8,ridx(18),ridx(17),ridx(19)) + 
    """
    // He++
    this_block_jacobian[%d] = -constants[%d]*ne;//He++ : -alpha_(He++)ne
    this_block_jacobian[%d] = -this_block_jacobian[%d];//He+ : 9-alpha_(He++)ne
    """%(ridx(24),9,ridx(23),ridx(24)))
    return jacobian_good_stuff

def get_derivative_block(this_tile,Ntile):
    ridx = lambda x: x+this_tile *5 
    derivative_good_stuff = ("""    // H0 : 2-alpha_(H+) ne nH+ - (0-Gamma_(e,H0)ne + 1-Gamma_(gamma,H0))*nH0
    this_block_derivatives[%d] = constants[%d]*ne*this_block_state[%d]
        -(constants[%d]*ne + constants[%d])*this_block_state[%d]; 
    """%(ridx(0),2,ridx(1),0,1,ridx(0)) )
    derivative_good_stuff+=(
    """
    // H+ : (Gamma_(e,H0)ne + Gamma_(gamma,H0))*nH0 - alpha_(H+) ne nH+
    this_block_derivatives[%d] = -this_block_derivatives[%d];
    """%(ridx(1),ridx(0)))
    derivative_good_stuff+=(
    """
    // He0 :(7-alpha_(He+)+8-alpha_(d)) ne 3-nHe+ - (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    this_block_derivatives[%d] = (constants[%d]+constants[%d])*ne*this_block_state[%d] 
        - (constants[%d]*ne+constants[%d])*this_block_state[%d];
    """%(ridx(2),7,8,ridx(3),3,4,ridx(2)))
    derivative_good_stuff+=(
    """
    // He+ : 
    //  9-alpha_(He++) ne nHe++ 
    //  + (3-Gamma_(e,He0)ne + 4-Gamma_(gamma,He0)) nHe0
    //  - (7-alpha_(He+)+8-alpha_(d)) ne nHe+ 
    //  - (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+
    this_block_derivatives[%d] = constants[%d]*ne*this_block_state[%d] 
        + (constants[%d]*ne+constants[%d])*this_block_state[%d]  
        - (constants[%d]+constants[%d])*ne*this_block_state[%d] 
        - (constants[%d]*ne+constants[%d])*this_block_state[%d];
    """%(ridx(3),9,ridx(4),3,4,ridx(2),7,8,ridx(3),5,6,ridx(3)))
    derivative_good_stuff+=(
    """
    // He++ : (5-Gamma_(e,He+)ne + 6-Gamma_(gamma,He+)) nHe+ 
    //  - 9-alpha_(He++) ne nHe++
    this_block_derivatives[%d] = (constants[%d]*ne+constants[%d])*this_block_state[%d]
        -constants[%d]*ne*this_block_state[%d]; 
    """%(ridx(4),5,6,ridx(3),9,ridx(4)))
    return derivative_good_stuff

def make_jacobian_string(Ntile):
    strr = jacobian_prefix
    for this_tile in range(Ntile):
        strr+=make_jacobian_block(this_tile,Ntile)
    return strr + jacobian_suffix


def make_derivative_string(Ntile):
    strr = derivative_prefix
    for this_tile in range(Ntile):
        strr+=get_derivative_block(this_tile,Ntile)
    return strr + derivative_suffix

def make_ode_file(Ntile):
    strr = file_prefix
    strr+=make_derivative_string(Ntile)
    strr+=make_jacobian_string(Ntile)
    strr+=file_suffix

    with open('preprocess_ode.cu','w') as handle:
        handle.write(strr)

    return make_ode_file
    

if __name__ == '__main__':
    arg = sys.argv[1:]
    if len(arg):
        arg = int(arg[0])
    else:
        arg = 1
    make_ode_file(arg)

