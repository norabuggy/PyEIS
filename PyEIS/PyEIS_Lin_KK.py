#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 4 18:23:35 2018

This script contains the core for the linear Kramer-Kronig analysis

@author: Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
"""
import numpy as np
from lmfit import Parameters

__all__ = ['KK_RC', 'residual_imag', 'residual_real', 'KK_Rnam_val', 'KK_timeconst', 'KK_errorfunc']


# Simulation
def KK_RC(w, Rs, R_values, t_values, num_RC):
    assert len(R_values) >= num_RC and len(t_values) >= num_RC
    R_values = R_values[:num_RC]
    t_values = t_values[:num_RC]
    return Rs + np.sum([r_val / (1 + w * 1j * t_val) for r_val, t_val in zip(R_values, t_values)],
                       axis=0)


# Fitting
def KK_RC_fit(params, w, t_values, num_RC):
    Rs = params['Rs']
    R_values = []
    for i in range(num_RC):
        R_values.append(params[f'R{i + 1}'])
    return KK_RC(w, Rs, R_values, t_values, num_RC)


# Least-squres function and related functions
def KK_Rnam_val(re, re_start, num_RC):
    """
    This function determines the name and initial guesses for resistances for the Linear KK test
    
    Ref.:
        - Schōnleber, M. et al. Electrochimica Acta 131 (2014) 20-27
        - Boukamp, B.A. J. Electrochem. Soc., 142, 6, 1885-1894 
        
    Kristian B. Knudsen (kknu@berkeley.edu || Kristianbknudsen@gmail.com)
    
    Inputs
    -----------
    w = angular frequency
    num_RC = number of -(RC)- circuits
    
    Outputs
    -----------
    [0] = parameters for LMfit
    [1] = R_names
    [2] = number of R in each fit
    """
    num_RC = np.arange(1, num_RC + 1, 1)

    R_name = []
    R_initial = []
    for j in range(len(num_RC)):
        R_name.append('R' + str(num_RC[j]))
        R_initial.append(1)  # initial guess for Resistances

    params = Parameters()
    for j in range(len(num_RC)):
        params.add(R_name[j], value=R_initial[j])

    params.add('Rs', value=re[re_start], min=-10 ** 5, max=10 ** 5)
    return params, R_name, num_RC


def KK_timeconst(w, num_RC):
    """
    This function determines the initial guesses for time constants for the Linear KK test
    
    Ref.:
        - Schōnleber, M. et al. Electrochimica Acta 131 (2014) 20-27
        
    Kristian B. Knudsen (kknu@berkeley.edu || Kristianbknudsen@gmail.com)
    """
    num_RC = np.arange(1, num_RC + 1, 1)

    t_max = 1 / min(w)
    t_min = 1 / max(w)
    t_name = []
    t_initial = []
    for j in range(len(num_RC)):
        t_name.append('t' + str(num_RC[j]))
        # initial guess parameter parameter tau for each -RC- circuit
        t_initial.append(
            10 ** ((np.log10(t_min)) + (j - 1) / (len(num_RC) - 1) * np.log10(t_max / t_min)))
    return t_initial


def KK_errorfunc(params, w, re, im, num_RC, weight_func, t_values):
    """
    Sum of squares error function for linear least-squares fitting for the Kramers-Kronig Relations. 
    The fitting function will use this function to iterate over until the return the sum of errors
     is minimized
    
    The data should be minimized using the weight_func = 'Boukamp'
    
    Ref.: Boukamp, B.A. J. Electrochem. Soc., 142, 6, 1885-1894 
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ------------
        - w = angular frequency
        - re = real impedance
        - im = imaginary impedance
        - num_RC = number of RC-circuits
        - t_values = time constants
        
        weight_func = Weight function, Three options:
            - modulus
            - unity
            - proportional
            - Boukamp
    """
    fit = KK_RC_fit(params, w, t_values, num_RC)
    re_fit = fit.real
    im_fit = -fit.imag
    # sum of squares
    error = [(re - re_fit) ** 2, (im - im_fit) ** 2]

    if weight_func == 'modulus':
        weight = [1 / ((re_fit ** 2 + im_fit ** 2) ** (1 / 2)),
                  1 / ((re_fit ** 2 + im_fit ** 2) ** (1 / 2))]
    elif weight_func == 'proportional':
        weight = [1 / (re_fit ** 2), 1 / (im_fit ** 2)]
    elif weight_func == 'unity':
        unity_1s = []
        for k in range(len(re)):
            # makes an array of [1]'s, so that the weighing is == 1 * sum of squares.
            unity_1s.append(1)
        weight = [unity_1s, unity_1s]
    elif weight_func == 'Boukamp':
        weight = [1 / (re ** 2), 1 / (im ** 2)]
    elif weight_func == 'ignore':
        print('weight ignored')
        return error
    # weighted sum of squares 
    S = np.array(weight) * error
    return S


# Evaluate Fit
def residual_real(re, fit_re, fit_im):
    """
    Relative Residuals as based on Boukamp's definition

    Ref.:
        - Boukamp, B.A. J. Electrochem. SoC., 142, 6, 1885-1894 
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    """
    modulus_fit = (fit_re ** 2 + fit_im ** 2) ** (1 / 2)
    return (re - fit_re) / modulus_fit


def residual_imag(im, fit_re, fit_im):
    """
    Relative Residuals as based on Boukamp's  definition

    Ref.:
        - Boukamp, B.A. J. Electrochem. SoC., 142, 6, 1885-1894 
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    """
    modulus_fit = (fit_re ** 2 + fit_im ** 2) ** (1 / 2)
    return (im - fit_im) / modulus_fit
