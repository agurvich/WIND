import numpy as np

from ode_systems.preprocess.preprocess import make_ode_file

class ODEBase(object):
    pass

    def validate(self):
        self.init_constants()
        self.init_equations()
        self.calculate_jacobian()
        self.calculate_derivative()
        self.calculate_eqmss()

    def init_constants(self):
        raise NotImplementedError
        
    def init_equations(self):
        raise NotImplementedError

    def calculate_jacobian(self):
        raise NotImplementedError

    def calculate_derivative(self):
        raise NotImplementedError

    def calculate_eqmss(self):
        raise NotImplementedError

    def calculate_eqmss():
        raise NotImplementedError

    def preprocess(self,Ntile):
        make_ode_file(self,Ntile)
        
