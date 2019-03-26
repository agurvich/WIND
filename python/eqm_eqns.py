import numpy as np

def get_eqm_abundances(nH,TEMP,y_helium):
    constants_dict = {}
    constants_dict['Gamma_(e,H0)'] = 5.85e-11 * np.sqrt(TEMP)/(1+np.sqrt(TEMP/1e5)) * np.exp(-157809.1/TEMP) * nH
    constants_dict['Gamma_(gamma,H0)'] = 4.4e-11
    constants_dict['alpha_(H+)'] = 8.4e-11 / np.sqrt(TEMP) * (TEMP/1e3)**-0.2 / (1+(TEMP/1e6)**0.7) * nH
    constants_dict['Gamma_(e,He0)'] =  2.38e-11 * np.sqrt(TEMP)/(1+np.sqrt(TEMP/1e5)) * np.exp(-285335.4/TEMP) * nH
    constants_dict['Gamma_(gamma,He0)'] = 3.7e-12
    constants_dict['Gamma_(e,He+)'] =  5.68e-12 * np.sqrt(TEMP)/(1+np.sqrt(TEMP/1e5)) * np.exp(-631515.0/TEMP) * nH
    constants_dict['Gamma_(gamma,He+)'] = 1.7e-14
    constants_dict['alpha_(He+)'] = 1.5e-10 * TEMP**-0.6353 * nH
    constants_dict['alpha_(d)'] =  (
        1.9e-3 * TEMP**-1.5 * 
        np.exp(-470000.0/TEMP) * 
        (1+0.3*np.exp(-94000.0/TEMP)) * nH )
    constants_dict['alpha_(He++)'] = 3.36e-10 * TEMP**-0.5 * (TEMP/1e3)**-0.2 / (1+(TEMP/1e6)**0.7) * nH

    for key in constants_dict:
        if 'alpha' in key or '(e' in key:
            constants_dict[key]/=nH


    densities = np.zeros(5)
    ne = nH
    for i in range(1000):
        ne = sub_get_densities(constants_dict,nH,TEMP,densities,ne,y_helium) 
        #if (not i%100):
            #print(densities)
    return densities/nH


def sub_get_densities(constants_dict,nH,TEMP,densities,ne,y_helium):
    """ from eqns 33 - 38 of Katz96"""
    
    ## H0
    densities[0] = (
        nH * constants_dict['alpha_(H+)']/
        (constants_dict['alpha_(H+)'] + 
            constants_dict['Gamma_(e,H0)'] + 
            constants_dict['Gamma_(gamma,H0)']/ne)
    )

    ## H+
    densities[1] = nH - densities[0]

    ## He+
    densities[3] = (
        nH * y_helium/4  / 
        (1 + (
            (constants_dict['alpha_(He+)'] + constants_dict['alpha_(d)']) / 
            (constants_dict['Gamma_(e,He0)']+constants_dict['Gamma_(gamma,He0)']/ne)
            )
        + (
            (constants_dict['Gamma_(e,He+)']+constants_dict['Gamma_(gamma,He+)']/ne)/
            constants_dict['alpha_(He++)']
            )
        )
    )

    ## He0
    densities[2] = (
        densities[3] * (constants_dict['alpha_(He+)']+constants_dict['alpha_(d)'])/
        (constants_dict['Gamma_(e,He0)']+constants_dict['Gamma_(gamma,He0)']/ne)
    )

    ## He++
    densities[4] = (
        densities[3] *
        (constants_dict['Gamma_(e,He+)']+constants_dict['Gamma_(gamma,He+)']/ne)/
        constants_dict['alpha_(He++)']
    )

    ## ne
    ne = densities[1] + densities[3] + densities[4]*2.0
    return ne
