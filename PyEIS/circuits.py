from typing import Optional

import mpmath as mp
import numpy as np
from numpy.core._multiarray_umath import tanh, sinh
from scipy.constants import codata
F = codata.physical_constants['Faraday constant'][0]
Rg = codata.physical_constants['molar gas constant'][0]


def sinh(x):
    """
    As numpy gives errors when sinh becomes very large, above 10^250, this functions is used
    instead of np/mp.sinh()
    """
    return (1 - np.exp(-2*x))/(2*np.exp(-x))


def coth(x):
    """
    As numpy gives errors when coth becomes very large, above 10^250, this functions is used
    instead of np/mp.coth()
    """
    return (1 + np.exp(-2*x))/(1 - np.exp(-2*x))


def elem_L(w, L):
    """
    Simulation Function: -L-
    Returns the impedance of an inductor

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ----------
    w = Angular frequency [1/s]
    L = Inductance [ohm * s]
    """
    return 1j*w*L


def elem_C(w,C):
    """
    Simulation Function: -C-

    Inputs
    ----------
    w = Angular frequency [1/s]
    C = Capacitance [F]
    """
    return 1/(C*(w*1j))


def elem_Q(w,Q,n):
    """
    Simulation Function: -Q-

    Inputs
    ----------
    w = Angular frequency [1/s]
    Q = Constant phase element [s^n/ohm]
    n = Constant phase elelment exponent [-]
    """
    return 1/(Q*(w*1j)**n)


def cir_RsC(w, Rs, C):
    """
    Simulation Function: -Rs-C-

    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series resistance [Ohm]
    C = Capacitance [F]
    """
    return Rs + 1/(C*(w*1j))


def cir_RsQ(w, Rs, Q, n):
    """
    Simulation Function: -Rs-Q-

    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series resistance [Ohm]
    Q = Constant phase element [s^n/ohm]
    n = Constant phase elelment exponent [-]
    """
    return Rs + 1/(Q*(w*1j)**n)


def cir_RQ(w, R: Optional[float] = None, Q: Optional[float] = None,
           n: Optional[float] = None, fs: Optional[float] = None):
    """
    Simulation Function: -RQ-
    Return the impedance of an Rs-RQ circuit. See details for RQ under cir_RQ_fit()

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)

    Inputs
    ----------
    w = Angular frequency [1/s]
    R = Resistance [Ohm]
    Q = Constant phase element [s^n/ohm]
    n = Constant phase elelment exponent [-]
    fs = Summit frequency of RQ circuit [Hz]
    """
    if R is None:
        R = (1/(Q*(2*np.pi*fs)**n))
    elif Q is None:
        Q = (1/(R*(2*np.pi*fs)**n))
    elif n is None:
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    return R / (1+R*Q*(w*1j)**n)


def cir_RsRQ(w, Rs='none', R='none', Q='none', n='none', fs='none'):
    """
    Simulation Function: -Rs-RQ-
    Return the impedance of an Rs-RQ circuit. See details for RQ under cir_RQ_fit()

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)

    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series resistance [Ohm]
    R = Resistance [Ohm]
    Q = Constant phase element [s^n/ohm]
    n = Constant phase elelment exponent [-]
    fs = Summit frequency of RQ circuit [Hz]
    """
    if R == 'none':
        R = (1/(Q*(2*np.pi*fs)**n))
    elif Q == 'none':
        Q = (1/(R*(2*np.pi*fs)**n))
    elif n == 'none':
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    return Rs + (R/(1+R*Q*(w*1j)**n))


def cir_RC(w, C='none', R='none', fs='none'):
    """
    Simulation Function: -RC-
    Returns the impedance of an RC circuit, using RQ definations where n=1. see cir_RQ() for details

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ----------
    w = Angular frequency [1/s]
    R = Resistance [Ohm]
    C = Capacitance [F]
    fs = Summit frequency of RC circuit [Hz]
    """
    return cir_RQ(w, R=R, Q=C, n=1, fs=fs)


def cir_RsRQRQ(w, Rs, R='none', Q='none', n='none', fs='none', R2='none', Q2='none', n2='none', fs2='none'):
    """
    Simulation Function: -Rs-RQ-RQ-
    Return the impedance of an Rs-RQ circuit. See details for RQ under cir_RQ_fit()

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series Resistance [Ohm]

    R = Resistance [Ohm]
    Q = Constant phase element [s^n/ohm]
    n = Constant phase element exponent [-]
    fs = Summit frequency of RQ circuit [Hz]

    R2 = Resistance [Ohm]
    Q2 = Constant phase element [s^n/ohm]
    n2 = Constant phase element exponent [-]
    fs2 = Summit frequency of RQ circuit [Hz]
    """
    if R == 'none':
        R = (1/(Q*(2*np.pi*fs)**n))
    elif Q == 'none':
        Q = (1/(R*(2*np.pi*fs)**n))
    elif n == 'none':
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))

    if R2 == 'none':
        R2 = (1/(Q2*(2*np.pi*fs2)**n2))
    elif Q2 == 'none':
        Q2 = (1/(R2*(2*np.pi*fs2)**n2))
    elif n2 == 'none':
        n2 = np.log(Q2*R2)/np.log(1/(2*np.pi*fs2))

    return Rs + (R/(1+R*Q*(w*1j)**n)) + (R2/(1+R2*Q2*(w*1j)**n2))


# L-Rs-RQ-RQ
def cir_LRsRQRQ(w, Rs, L,
                  R1: Optional[float] = None, Q1: Optional[float] = None,
                  n1: Optional[float] = None, fs1: Optional[float] = None,
                  R2: Optional[float] = None, Q2: Optional[float] = None,
                  n2: Optional[float] = None, fs2: Optional[float] = None,):
    """
    Simulation Function: L-Rs-RQ-RQ-RQ


    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series Resistance [Ohm]
    L = Series Impedence

    R1 = Resistance [Ohm]
    Q1 = Constant phase element [s^n/ohm]
    n1 = Constant phase element exponent [-]
    fs1 = Summit frequency of RQ circuit [Hz]

    R2 = Resistance [Ohm]
    Q2 = Constant phase element [s^n/ohm]
    n2 = Constant phase element exponent [-]
    fs2 = Summit frequency of RQ circuit [Hz]
    """
    return elem_L(w, L) + Rs + cir_RQ(w, R1, Q1, n1, fs1) + cir_RQ(w, R2, Q2, n2, fs2)


def cir_LRsRQRQ_fit(params, w):
    return cir_LRsRQRQ(w, params['Rs'], params['L'],
                       R1=params.get('R1'), Q1=params.get('Q1'),
                       n1=params.get('n1'), fs1=params.get('fs1'),
                       R2=params.get('R2'), Q2=params.get('Q2'),
                       n2=params.get('n2'), fs2=params.get('fs2'))


# L-Rs-RQ-RQ-RQ
def cir_LRsRQRQRQ(w, Rs, L,
                  R1: Optional[float] = None, Q1: Optional[float] = None,
                  n1: Optional[float] = None, fs1: Optional[float] = None,
                  R2: Optional[float] = None, Q2: Optional[float] = None,
                  n2: Optional[float] = None, fs2: Optional[float] = None,
                  R3: Optional[float] = None, Q3: Optional[float] = None,
                  n3: Optional[float] = None, fs3: Optional[float] = None,):
    """
    Simulation Function: L-Rs-RQ-RQ-RQ


    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series Resistance [Ohm]
    L = Series Impedence

    R1 = Resistance [Ohm]
    Q1 = Constant phase element [s^n/ohm]
    n1 = Constant phase element exponent [-]
    fs1 = Summit frequency of RQ circuit [Hz]

    R2 = Resistance [Ohm]
    Q2 = Constant phase element [s^n/ohm]
    n2 = Constant phase element exponent [-]
    fs2 = Summit frequency of RQ circuit [Hz]

    R3 = Resistance [Ohm]
    Q3 = Constant phase element [s^n/ohm]
    n3 = Constant phase element exponent [-]
    fs3 = Summit frequency of RQ circuit [Hz]
    """
    return elem_L(w, L) + Rs + cir_RQ(w, R1, Q1, n1, fs1) + \
                               cir_RQ(w, R2, Q2, n2, fs2) + \
                               cir_RQ(w, R3, Q3, n3, fs3)


def cir_LRsRQRQRQ_fit(params, w):
    return cir_LRsRQRQRQ(w, params['Rs'], params['L'],
                         R1=params.get('R1'), Q1=params.get('Q1'),
                         n1=params.get('n1'), fs1=params.get('fs1'),
                         R2=params.get('R2'), Q2=params.get('Q2'),
                         n2=params.get('n2'), fs2=params.get('fs2'),
                         R3=params.get('R3'), Q3=params.get('Q3'),
                         n3=params.get('n3'), fs3=params.get('fs3'))


#  L-R-Q(R)-Q(R-Q(R))-Q(R)
def cir_LRQRQRQRQR(w, Rs, L,
                   R1=None, Q1=None, n1=None, fs1=None,
                   R2=None, Q2=None, n2=None,
                   R3=None, Q3=None, n3=None, fs3=None,
                   R4=None, Q4=None, n4=None, fs4=None):
    mid_element = (1/elem_Q(w, Q2, n2) + 1/(R2 + cir_RQ(w, R3, Q3, n3, fs3)))**-1
    return elem_L(w, L) + Rs + cir_RQ(w, R1, Q1, n1, fs1) + mid_element + cir_RQ(w, R4, Q4, n4, fs4)


def cir_LRQRQRQRQR_fit(params, w):
    return cir_LRQRQRQRQR(w, params['Rs'], params['L'],
                          R1=params.get('R1'), Q1=params.get('Q1'),
                          n1=params.get('n1'), fs1=params.get('fs1'),
                          R2=params.get('R2'), Q2=params.get('Q2'),
                          n2=params.get('n2'),
                          R3=params.get('R3'), Q3=params.get('Q3'),
                          n3=params.get('n3'), fs3=params.get('fs3'),
                          R4=params.get('R4'), Q4=params.get('Q4'),
                          n4=params.get('n4'), fs4=params.get('fs4'))


def cir_RsRQQ(w, Rs, Q, n, R1='none', Q1='none', n1='none', fs1='none'):
    """
    Simulation Function: -Rs-RQ-Q-

    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series Resistance [ohm]

    R1 = Resistance in (RQ) circuit [ohm]
    Q1 = Constant phase element in (RQ) circuit [s^n/ohm]
    n1 = Constant phase elelment exponent in (RQ) circuit [-]
    fs1 = Summit frequency of RQ circuit [Hz]

    Q = Constant phase element of series Q [s^n/ohm]
    n = Constant phase elelment exponent of series Q [-]
    """
    return Rs + cir_RQ(w, R=R1, Q=Q1, n=n1, fs=fs1) + elem_Q(w,Q,n)


def cir_RsRQC(w, Rs, C, R1='none', Q1='none', n1='none', fs1='none'):
    """
    Simulation Function: -Rs-RQ-C-

    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series Resistance [ohm]

    R1 = Resistance in (RQ) circuit [ohm]
    Q1 = Constant phase element in (RQ) circuit [s^n/ohm]
    n1 = Constant phase elelment exponent in (RQ) circuit [-]
    fs1 = summit frequency of RQ circuit [Hz]

    C = Constant phase element of series Q [s^n/ohm]
    """
    return Rs + cir_RQ(w, R=R1, Q=Q1, n=n1, fs=fs1) + elem_C(w, C=C)


def cir_RsRCC(w, Rs, R1, C1, C):
    """
    Simulation Function: -Rs-RC-C-

    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series Resistance [ohm]

    R1 = Resistance in (RQ) circuit [ohm]
    C1 = Constant phase element in (RQ) circuit [s^n/ohm]

    C = Capacitance of series C [s^n/ohm]
    """
    return Rs + cir_RC(w, C=C1, R=R1, fs='none') + elem_C(w, C=C)


def cir_RsRCQ(w, Rs, R1, C1, Q, n):
    """
    Simulation Function: -Rs-RC-Q-

    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series Resistance [ohm]

    R1 = Resistance in (RQ) circuit [ohm]
    C1 = Constant phase element in (RQ) circuit [s^n/ohm]

    Q = Constant phase element of series Q [s^n/ohm]
    n = Constant phase elelment exponent of series Q [-]
    """
    return Rs + cir_RC(w, C=C1, R=R1, fs='none') + elem_Q(w,Q,n)


def Randles_coeff(w, n_electron, A, E='none', E0='none', D_red='none', D_ox='none', C_red='none', C_ox='none', Rg=Rg, F=F, T=298.15):
    """
    Returns the Randles coefficient sigma [ohm/s^1/2].
    Two cases: a) ox and red are both present in solution here both Cred and Dred are defined, b) In the particular case where initially
    only Ox species are present in the solution with bulk concentration C*_ox, the surface concentrations may be calculated as function
    of the electrode potential following Nernst equation. Here C_red and D_red == 'none'

    Ref.:
        - Lasia, A.L., ISBN: 978-1-4614-8932-0, "Electrochemical Impedance Spectroscopy and its Applications"
        - Bard A.J., ISBN: 0-471-04372-9, Faulkner L. R. (2001) "Electrochemical methods: Fundamentals and applications". New York: Wiley.

    Kristian B. Knudsen (kknu@berkeley.edu // kristianbknudsen@gmail.com)

    Inputs
    ----------
    n_electron = number of e- [-]
    A = geometrical surface area [cm2]
    D_ox = Diffusion coefficent of oxidized specie [cm2/s]
    D_red = Diffusion coefficent of reduced specie [cm2/s]
    C_ox = Bulk concetration of oxidized specie [mol/cm3]
    C_red = Bulk concetration of reduced specie [mol/cm3]
    T = Temperature [K]
    Rg = Gas constant [J/molK]
    F = Faradays consntat [C/mol]
    E = Potential [V]
        if reduced specie is absent == 'none'
    E0 = formal potential [V]
        if reduced specie is absent == 'none'

    Returns
    ----------
    Randles coefficient [ohm/s^1/2]
    """
    if C_red != 'none' and D_red != 'none':
        sigma = ((Rg*T) / ((n_electron**2) * A * (F**2) * (2**(1/2)))) * ((1/(D_ox**(1/2) * C_ox)) + (1/(D_red**(1/2) * C_red)))
    elif C_red == 'none' and D_red == 'none' and E!='none' and E0!= 'none':
        f = F/(Rg*T)
        x = (n_electron*f*(E-E0))/2
        func_cosh2 = (np.cosh(2*x)+1)/2
        sigma = ((4*Rg*T) / ((n_electron**2) * A * (F**2) * C_ox * ((2*D_ox)**(1/2)) )) * func_cosh2
    else:
        print('define E and E0')
    Z_Aw = sigma*(w**(-0.5))-1j*sigma*(w**(-0.5))
    return Z_Aw


def cir_Randles(w, n_electron, D_red, D_ox, C_red, C_ox, Rs, Rct, n, E, A, Q='none', fs='none', E0=0, F=F, Rg=Rg, T=298.15):
    """
    Simulation Function: Randles -Rs-(Q-(RW)-)-
    Return the impedance of a Randles circuit with full complity of the warbug constant
    NOTE: This Randles circuit is only meant for semi-infinate linear diffusion

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)

    Inputs
    ----------
    n_electron = number of e- [-]
    A = geometrical surface area [cm2]
    D_ox = Diffusion coefficent of oxidized specie [cm2/s]
    D_red = Diffusion coefficent of reduced specie [cm2/s]
    C_ox = Concetration of oxidized specie [mol/cm3]
    C_red = Concetration of reduced specie [mol/cm3]
    T = Temperature [K]
    Rg = Gas constant [J/molK]
    F = Faradays consntat [C/mol]
    E = Potential [V]
        if reduced specie is absent == 'none'
    E0 = Formal potential [V]
        if reduced specie is absent == 'none'

    Rs = Series resistance [ohm]
    Rct = charge-transfer resistance [ohm]

    Q = Constant phase element used to model the double-layer capacitance [F]
    n = expononent of the CPE [-]

    Returns
    ----------
    The real and imaginary impedance of a Randles circuit [ohm]
    """
    Z_Rct = Rct
    Z_Q = elem_Q(w,Q,n)
    Z_w = Randles_coeff(w, n_electron=n_electron, E=E, E0=E0, D_red=D_red, D_ox=D_ox, C_red=C_red, C_ox=C_ox, A=A, T=T, Rg=Rg, F=F)
    return Rs + 1/(1/Z_Q + 1/(Z_Rct+Z_w))


def cir_Randles_simplified(w, Rs, R, n, sigma, Q='none', fs='none'):
    """
    Simulation Function: Randles -Rs-(Q-(RW)-)-
    Return the impedance of a Randles circuit with a simplified

    NOTE: This Randles circuit is only meant for semi-infinate linear diffusion

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    """
    if R == 'none':
        R = (1/(Q*(2*np.pi*fs)**n))
    elif Q == 'none':
        Q = (1/(R*(2*np.pi*fs)**n))
    elif n == 'none':
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))

    Z_Q = 1/(Q*(w*1j)**n)
    Z_R = R
    Z_w = sigma*(w**(-0.5))-1j*sigma*(w**(-0.5))

    return Rs + 1/(1/Z_Q + 1/(Z_R+Z_w))


def cir_C_RC_C(w, Ce, Cb='none', Rb='none', fsb='none'):
    """
    Simulation Function: -C-(RC)-C-

    This circuit is often used for modeling blocking electrodes with a polymeric electrolyte, which
    exhibts a immobile ionic species in bulk that gives a capacitance contribution
    to the otherwise resistive electrolyte

    Ref:
    - MacCallum J.R., and Vincent, C.A. "Polymer Electrolyte Reviews - 1" Elsevier Applied Science
     Publishers LTD, London, Bruce, P. "Electrical Measurements on Polymer Electrolytes"

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ----------
    w = Angular frequency [1/s]
    Ce = Interfacial capacitance [F]
    Rb = Bulk/series resistance [Ohm]
    Cb = Bulk capacitance [F]
    fsb = summit frequency of bulk (RC) circuit [Hz]
    """
    Z_C = elem_C(w,C=Ce)
    Z_RC = cir_RC(w, C=Cb, R=Rb, fs=fsb)
    return Z_C + Z_RC


def cir_Q_RQ_Q(w, Qe, ne, Qb='none', Rb='none', fsb='none', nb='none'):
    """
    Simulation Function: -Q-(RQ)-Q-

    Modified cir_C_RC_C() circuits that can be used if electrodes and bulk are not behaving like
    ideal capacitors

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ----------
    w = Angular frequency [1/s]
    Qe = Interfacial capacitance modeled with a CPE [F]
    ne = Interfacial constant phase element exponent [-]

    Rb = Bulk/series resistance [Ohm]
    Qb = Bulk capacitance modeled with a CPE [s^n/ohm]
    nb = Bulk constant phase element exponent [-]
    fsb = summit frequency of bulk (RQ) circuit [Hz]
    """
    Z_Q = elem_Q(w,Q=Qe,n=ne)
    Z_RQ = cir_RQ(w, Q=Qb, R=Rb, fs=fsb, n=nb)
    return Z_Q + Z_RQ


def tanh(x):
    """
    As numpy gives errors when tanh becomes very large, above 10^250, this functions is used for np.tanh
    """
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))


def cir_RCRCZD(w, L, D_s, u1, u2, Cb='none', Rb='none', fsb='none', Ce='none', Re='none', fse='none'):
    """
    Simulation Function: -RC_b-RC_e-Z_D

    This circuit has been used to study non-blocking electrodes with an ioniocally conducting
    electrolyte with a mobile and immobile ionic specie in bulk, this is mixed with a
    ionically conducting salt. This behavior yields in a impedance response, that consists of the
    interfacial impendaces -(RC_e)-, the ionically conducitng polymer -(RC_e)-,
    and the diffusional impedance from the dissolved salt.

    Refs.:
        - Sørensen, P.R. and Jacobsen T., Electrochimica Acta, 27, 1671-1675, 1982, "Conductivity,
        Charge Transfer and Transport number - An AC-Investigation
        of the Polymer Electrolyte LiSCN-Poly(ethyleneoxide)"
        - MacCallum J.R., and Vincent, C.A. "Polymer Electrolyte Reviews - 1" Elsevier Applied
        Science Publishers LTD, London
        Bruce, P. "Electrical Measurements on Polymer Electrolytes"

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ----------
    w = Angular frequency [1/s]
    L = Thickness of electrode [cm]
    D_s = Diffusion coefficient of dissolved salt [cm2/s]
    u1 = Mobility of the ion reacting at the electrode interface
    u2 = Mobility of other ion

    Re = Interfacial resistance [Ohm]
    Ce = Interfacial  capacitance [F]
    fse = Summit frequency of the interfacial (RC) circuit [Hz]

    Rb = Bulk/series resistance [Ohm]
    Cb = Bulk capacitance [F]
    fsb = Summit frequency of the bulk (RC) circuit [Hz]
    """
    Z_RCb = cir_RC(w, C=Cb, R=Rb, fs=fsb)
    Z_RCe = cir_RC(w, C=Ce, R=Re, fs=fse)
    alpha = ((w*1j*L**2)/D_s)**(1/2)
    Z_D = Rb * (u2/u1) * (tanh(x=alpha)/alpha)
    return Z_RCb + Z_RCe + Z_D


def cir_RsTLsQ(w, Rs, L, Ri, Q='none', n='none'):
    """
    Simulation Function: -Rs-TLsQ-
    TLs = Simplified Transmission Line, with a non-faradaic interfacial impedance (Q)

    The simplified transmission line assumes that Ri is much greater than Rel (electrode resistance).

    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering,
         p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects
         in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the
        analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)

    Inputs
    -----------
    Rs = Series resistance [ohm]

    L = Length/Thickness of porous electrode [cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    Q = Interfacial capacitance of non-faradaic interface [F/cm]
    n = exponent for the interfacial capacitance [-]
    """
    Phi = 1/(Q*(w*1j)**n)
    X1 = Ri # ohm/cm
    Lam = (Phi/X1)**(1/2) #np.sqrt(Phi/X1)

    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)  #Handles coth with x having very large or very small numbers

    Z_TLsQ = Lam * X1 * coth_mp

    return Rs + Z_TLsQ


def cir_RsRQTLsQ(w, Rs, R1, fs1, n1, L, Ri, Q, n, Q1='none'):
    """
    Simulation Function: -Rs-RQ-TLsQ-
    TLs = Simplified Transmission Line, with a non-faradaic interfacial impedance(Q)

    The simplified transmission line assumes that Ri is much greater than Rel (electrode resistance).

    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering,
         p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects
         in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the
         analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)

    Inputs
    -----------
    Rs = Series resistance [ohm]

    R1 = Charge transfer resistance of RQ circuit [ohm]
    fs1 = Summit frequency for RQ circuit [Hz]
    n1 = Exponent for RQ circuit [-]
    Q1 = Constant phase element of RQ circuit [s^n/ohm]

    L = Length/Thickness of porous electrode [cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    Q = Interfacial capacitance of non-faradaic interface [F/cm]
    n = Exponent for the interfacial capacitance [-]

    Output
    -----------
    Impdance of Rs-(RQ)1-TLsQ
    """
    Z_RQ = cir_RQ(w=w, R=R1, Q=Q1, n=n1, fs=fs1)

    Phi = 1/(Q*(w*1j)**n)
    X1 = Ri
    Lam = (Phi/X1)**(1/2)

    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)

    Z_TLsQ = Lam * X1 * coth_mp

    return Rs + Z_RQ + Z_TLsQ


def cir_RsTLs(w, Rs, L, Ri, R='none', Q='none', n='none', fs='none'):
    """
    Simulation Function: -Rs-TLs-
    TLs = Simplified Transmission Line, with a faradaic interfacial impedance (RQ)

    The simplified transmission line assumes that Ri is much greater than Rel (electrode resistance).

    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering,
         p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects
        in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the
        analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)

    Inputs
    -----------
    Rs = Series resistance [ohm]

    L = Length/Thickness of porous electrode [cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    R = Interfacial Charge transfer resistance [ohm*cm]
    fs = Summit frequency of interfacial RQ circuit [Hz]
    n = Exponent for interfacial RQ circuit [-]
    Q = Constant phase element of interfacial capacitance [s^n/Ohm]

    Output
    -----------
    Impedance of Rs-TLs(RQ)
    """
    Phi = cir_RQ(w, R, Q, n, fs)
    X1 = Ri
    Lam = (Phi/X1)**(1/2)

    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers

    Z_TLs = Lam * X1 * coth_mp

    return Rs + Z_TLs


def cir_RsRQTLs(w, Rs, L, Ri, R1, n1, fs1, R2, n2, fs2, Q1='none', Q2='none'):
    """
    Simulation Function: -Rs-RQ-TLs-
    TLs = Simplified Transmission Line, with a faradaic interfacial impedance (RQ)

    The simplified transmission line assumes that Ri is much greater than Rel (electrode resistance).

    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)

    Inputs
    -----------
    Rs = Series resistance [ohm]

    R1 = Charge transfer resistance of RQ circuit [ohm]
    fs1 = Summit frequency for RQ circuit [Hz]
    n1 = Exponent for RQ circuit [-]
    Q1 = Constant phase element of RQ circuit [s^n/(ohm * cm)]

    L = Length/Thickness of porous electrode [cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    R2 = Interfacial Charge transfer resistance [ohm*cm]
    fs2 = Summit frequency of interfacial RQ circuit [Hz]
    n2 = Exponent for interfacial RQ circuit [-]
    Q2 = Constant phase element of interfacial capacitance [s^n/Ohm]

    Output
    -----------
    Impedance of Rs-(RQ)1-TLs(RQ)2
    """
    Z_RQ = cir_RQ(w=w, R=R1, Q=Q1, n=n1, fs=fs1)

    Phi = cir_RQ(w=w, R=R2, Q=Q2, n=n2, fs=fs2)
    X1 = Ri
    Lam = (Phi/X1)**(1/2)

    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers

    Z_TLs = Lam * X1 * coth_mp

    return Rs + Z_RQ + Z_TLs




def cir_RsTLQ(w, L, Rs, Q, n, Rel, Ri):
    """
    Simulation Function: -R-TLQ- (interfacial non-reacting, i.e. blocking electrode)
    Transmission line w/ full complexity, which both includes Ri and Rel

    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ------------------
    Rs = Series resistance [ohm]

    Q = Constant phase element for the interfacial capacitance [s^n/ohm]
    n = exponenet for interfacial RQ element [-]

    Rel = electronic resistance of electrode [ohm/cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    L = thickness of porous electrode [cm]

    Output
    --------------
    Impedance of Rs-TL
    """
    #The impedance of the series resistance
    Z_Rs = Rs

    # The Interfacial impedance is given by an -(RQ)- circuit
    Phi = elem_Q(w, Q=Q, n=n)
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_TL


def cir_RsRQTLQ(w, L, Rs, Q, n, Rel, Ri, R1, n1, fs1, Q1='none'):
    """
    Simulation Function: -R-RQ-TLQ- (interfacial non-reacting, i.e. blocking electrode)
    Transmission line w/ full complexity, which both includes Ri and Rel

    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ------------------
    Rs = Series resistance [ohm]

    R1 = Charge transfer resistance of RQ circuit [ohm]
    fs1 = Summit frequency for RQ circuit [Hz]
    n1 = exponent for RQ circuit [-]
    Q1 = constant phase element of RQ circuit [s^n/(ohm * cm)]

    Q = Constant phase element for the interfacial capacitance [s^n/ohm]
    n = exponenet for interfacial RQ element [-]

    Rel = electronic resistance of electrode [ohm/cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    L = thickness of porous electrode [cm]

    Output
    --------------
    Impedance of Rs-TL
    """
    #The impedance of the series resistance
    Z_Rs = Rs

    #The (RQ) circuit in series with the transmission line
    Z_RQ1 = cir_RQ(w=w, R=R1, Q=Q1, n=n1, fs=fs1)

    # The Interfacial impedance is given by an -(RQ)- circuit
    Phi = elem_Q(w, Q=Q, n=n)
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_RQ1 + Z_TL


def cir_RsTL(w, L, Rs, R, fs, n, Rel, Ri, Q='none'):
    """
    Simulation Function: -R-TL- (interfacial reacting, i.e. non-blocking)
    Transmission line w/ full complexity, which both includes Ri and Rel

    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ------------------
    Rs = Series resistance [ohm]

    R = Interfacial charge transfer resistance [ohm * cm]
    fs = Summit frequency for the interfacial RQ element [Hz]
    n = Exponenet for interfacial RQ element [-]
    Q = Constant phase element for the interfacial capacitance [s^n/ohm]

    Rel = Electronic resistance of electrode [ohm/cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    L = Thickness of porous electrode [cm]

    Output
    --------------
    Impedance of Rs-TL
    """
    #The impedance of the series resistance
    Z_Rs = Rs

    # The Interfacial impedance is given by an -(RQ)- circuit
    Phi = cir_RQ(w, R=R, Q=Q, n=n, fs=fs)
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_TL


def cir_RsRQTL(w, L, Rs, R1, fs1, n1, R2, fs2, n2, Rel, Ri, Q1='none', Q2='none'):
    """
    Simulation Function: -R-RQ-TL- (interfacial reacting, i.e. non-blocking)
    Transmission line w/ full complexity, which both includes Ri and Rel

    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ------------------
    Rs = Series resistance [ohm]

    R1 = Charge transfer resistance of RQ circuit [ohm]
    fs1 = Summit frequency for RQ circuit [Hz]
    n1 = exponent for RQ circuit [-]
    Q1 = constant phase element of RQ circuit [s^n/(ohm * cm)]

    R2 = interfacial charge transfer resistance [ohm * cm]
    fs2 = Summit frequency for the interfacial RQ element [Hz]
    n2 = exponenet for interfacial RQ element [-]
    Q2 = Constant phase element for the interfacial capacitance [s^n/ohm]

    Rel = electronic resistance of electrode [ohm/cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    L = thickness of porous electrode [cm]

    Output
    --------------
    Impedance of Rs-TL
    """
    #The impedance of the series resistance
    Z_Rs = Rs

    #The (RQ) circuit in series with the transmission line
    Z_RQ1 = cir_RQ(w=w, R=R1, Q=Q1, n=n1, fs=fs1)

    # The Interfacial impedance is given by an -(RQ)- circuit
    Phi = cir_RQ(w, R=R2, Q=Q2, n=n2, fs=fs2)
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)
    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)
    return Z_Rs + Z_RQ1 + Z_TL


def cir_RsTL_1Dsolid(w, L, D, radius, Rs, R, Q, n, R_w, n_w, Rel, Ri):
    """
    Simulation Function: -R-TL(Q(RW))-
    Transmission line w/ full complexity, which both includes Ri and Rel

    Warburg element is specific for 1D solid-state diffusion

    Refs:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Illig, J., Physically based Impedance Modelling of Lithium-ion Cells, KIT Scientific Publishing (2014)
        - Scipioni, et al., ECS Transactions, 69 (18) 71-80 (2015)

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ------------------
    Rs = Series resistance [ohm]

    R = particle charge transfer resistance [ohm*cm^2]
    Q = Summit frequency peak of RQ element in the modified randles element of a particle [Hz]
    n = exponenet for internal RQ element in the modified randles element of a particle [-]

    Rel = electronic resistance of electrode [ohm/cm]
    Ri = ionic resistance of solution in flooded pores of electrode [ohm/cm]
    R_w = polarization resistance of finite diffusion Warburg element [ohm]
    n_w = exponent for Warburg element [-]

    L = thickness of porous electrode [cm]
    D = solid-state diffusion coefficient [cm^2/s]
    radius = average particle radius [cm]

    Output
    --------------
    Impedance of Rs-TL(Q(RW))
    """
    #The impedance of the series resistance
    Z_Rs = Rs

    #The impedance of a 1D Warburg Element
    time_const = (radius**2)/D

    x = (time_const*w*1j)**n_w
    x_mp = mp.matrix(x)
    warburg_coth_mp = []
    for i in range(len(w)):
        warburg_coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)

    Z_w = R_w * np.array(warburg_coth_mp)/x

    # The Interfacial impedance is given by a Randles Equivalent circuit with the finite space warburg element in series with R2
    Z_Rct = R
    Z_Q = elem_Q(w,Q=Q,n=n)
    Z_Randles = 1/(1/Z_Q + 1/(Z_Rct+Z_w)) #Ohm

    # The Impedance of the Transmission Line
    lamb = (Z_Randles/(Rel+Ri))**(1/2)
    x = L/lamb
#    lamb_mp = mp.matrix(x)
#    sinh_mp = []
#    coth_mp = []
#    for j in range(len(lamb_mp)):
#        sinh_mp.append(float(mp.sinh(lamb_mp[j]).real)+float((mp.sinh(lamb_mp[j]).imag))*1j)
#        coth_mp.append(float(mp.coth(lamb_mp[j]).real)+float(mp.coth(lamb_mp[j]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/np.array(sinh_mp))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)
    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/sinh(x))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)
    return Z_Rs + Z_TL


def cir_RsRQTL_1Dsolid(w, L, D, radius, Rs, R1, fs1, n1, R2, Q2, n2, R_w, n_w, Rel, Ri, Q1='none'):
    """
    Simulation Function: -R-RQ-TL(Q(RW))-
    Transmission line w/ full complexity, which both includes Ri and Rel

    Warburg element is specific for 1D solid-state diffusion

    Refs:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"
        - Illig, J., Physically based Impedance Modelling of Lithium-ion Cells, KIT Scientific Publishing (2014)
        - Scipioni, et al., ECS Transactions, 69 (18) 71-80 (2015)

    David Brown (demoryb@berkeley.edu)
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ------------------
    Rs = Series resistance [ohm]

    R1 = charge transfer resistance of the interfacial RQ element [ohm*cm^2]
    fs1 = max frequency peak of the interfacial RQ element[Hz]
    n1 = exponenet for interfacial RQ element

    R2 = particle charge transfer resistance [ohm*cm^2]
    Q2 = Summit frequency peak of RQ element in the modified randles element of a particle [Hz]
    n2 = exponenet for internal RQ element in the modified randles element of a particle [-]

    Rel = electronic resistance of electrode [ohm/cm]
    Ri = ionic resistance of solution in flooded pores of electrode [ohm/cm]
    R_w = polarization resistance of finite diffusion Warburg element [ohm]
    n_w = exponent for Warburg element [-]

    L = thickness of porous electrode [cm]
    D = solid-state diffusion coefficient [cm^2/s]
    radius = average particle radius [cm]

    Output
    ------------------
    Impedance of R-RQ-TL(Q(RW))
    """
    #The impedance of the series resistance
    Z_Rs = Rs

    # The Interfacial impedance is given by an -(RQ)- circuit
    Z_RQ = cir_RQ(w=w, R=R1, Q=Q1, n=n1, fs=fs1)

    #The impedance of a 1D Warburg Element
    time_const = (radius**2)/D

    x = (time_const*w*1j)**n_w
    x_mp = mp.matrix(x)
    warburg_coth_mp = []
    for i in range(len(w)):
        warburg_coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)

    Z_w = R_w * np.array(warburg_coth_mp)/x

    # The Interfacial impedance is given by a Randles Equivalent circuit with the finite space warburg element in series with R2
    Z_Rct = R2
    Z_Q = elem_Q(w,Q=Q2,n=n2)
    Z_Randles = 1/(1/Z_Q + 1/(Z_Rct+Z_w)) #Ohm

    # The Impedance of the Transmission Line
    lamb = (Z_Randles/(Rel+Ri))**(1/2)
    x = L/lamb
#    lamb_mp = mp.matrix(x)
#    sinh_mp = []
#    coth_mp = []
#    for j in range(len(lamb_mp)):
#        sinh_mp.append(float(mp.sinh(lamb_mp[j]).real)+float((mp.sinh(lamb_mp[j]).imag))*1j)
#        coth_mp.append(float(mp.coth(lamb_mp[j]).real)+float(mp.coth(lamb_mp[j]).imag)*1j)
#
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/np.array(sinh_mp))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/sinh(x))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_RQ + Z_TL


###################################### FITS ####################################################
def elem_C_fit(params, w):
    """
    Fit Function: -C-
    """
    C = params['C']
    return 1/(C*(w*1j))


def elem_Q_fit(params, w):
    """
    Fit Function: -Q-

    Constant Phase Element for Fitting
    """
    Q = params['Q']
    n = params['n']
    return 1/(Q*(w*1j)**n)


def cir_RsC_fit(params, w):
    """
    Fit Function: -Rs-C-
    """
    Rs = params['Rs']
    C = params['C']
    return Rs + 1/(C*(w*1j))


def cir_RsQ_fit(params, w):
    """
    Fit Function: -Rs-Q-
    """
    Rs = params['Rs']
    Q = params['Q']
    n = params['n']
    return Rs + 1/(Q*(w*1j)**n)


def cir_RC_fit(params, w):
    """
    Fit Function: -RC-
    Returns the impedance of an RC circuit, using RQ definitions where n=1
    """
    n=1
    if str(params.keys())[10:].find("R") == -1: #if R == 'none':
        Q = params['C']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("C") == -1: #elif Q == 'none':
        R = params['R']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    # if str(params.keys())[10:].find("n") == -1: #elif n == 'none':
    #     R = params['R']
    #     Q = params['C']
    #     fs = params['fs']
    #     n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("fs") == -1: #elif fs == 'none':
        R = params['R']
        Q = params['C']
    return cir_RQ(w, R=R, Q=Q, n=1, fs=fs)


def cir_RQ_fit(params, w):
    """
    Fit Function: -RQ-
    Return the impedance of an RQ circuit:
    Z(w) = R / (1+ R*Q * (2w)^n)

    See Explanation of equations under cir_RQ()

    The params.keys()[10:] finds the names of the user defined parameters that should be interated
    over if X == -1, if the paramter is not given, it becomes equal to 'none'

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    """
    if str(params.keys())[10:].find("R") == -1: #if R == 'none':
        Q = params['Q']
        n = params['n']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("Q") == -1: #elif Q == 'none':
        R = params['R']
        n = params['n']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("n") == -1: #elif n == 'none':
        R = params['R']
        Q = params['Q']
        fs = params['fs']
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("fs") == -1: #elif fs == 'none':
        R = params['R']
        n = params['n']
        Q = params['Q']
    return R/(1+R*Q*(w*1j)**n)


def cir_RsRQ_fit(params, w):
    """
    Fit Function: -Rs-RQ-
    Return the impedance of an Rs-RQ circuit. See details for RQ under cir_RsRQ_fit()

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    """
    if str(params.keys())[10:].find("R") == -1: #if R == 'none':
        Q = params['Q']
        n = params['n']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("Q") == -1: #elif Q == 'none':
        R = params['R']
        n = params['n']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("n") == -1: #elif n == 'none':
        R = params['R']
        Q = params['Q']
        fs = params['fs']
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("fs") == -1: #elif fs == 'none':
        R = params['R']
        Q = params['Q']
        n = params['n']

    Rs = params['Rs']
    return Rs + (R/(1+R*Q*(w*1j)**n))


def cir_RsRQRQ_fit(params, w):
    """
    Fit Function: -Rs-RQ-RQ-
    Return the impedance of an Rs-RQ circuit. See details under cir_RsRQRQ()

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    """
    if str(params.keys())[10:].find("'R'") == -1: #if R == 'none':
        Q = params['Q']
        n = params['n']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("'Q'") == -1: #elif Q == 'none':
        R = params['R']
        n = params['n']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("'n'") == -1: #elif n == 'none':
        R = params['R']
        Q = params['Q']
        fs = params['fs']
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("'fs'") == -1: #elif fs == 'none':
        R = params['R']
        Q = params['Q']
        n = params['n']

    if str(params.keys())[10:].find("'R2'") == -1: #if R == 'none':
        Q2 = params['Q2']
        n2 = params['n2']
        fs2 = params['fs2']
        R2 = (1/(Q2*(2*np.pi*fs2)**n2))
    if str(params.keys())[10:].find("'Q2'") == -1: #elif Q == 'none':
        R2 = params['R2']
        n2 = params['n2']
        fs2 = params['fs2']
        Q2 = (1/(R2*(2*np.pi*fs2)**n2))
    if str(params.keys())[10:].find("'n2'") == -1: #elif n == 'none':
        R2 = params['R2']
        Q2 = params['Q2']
        fs2 = params['fs2']
        n2 = np.log(Q2*R2)/np.log(1/(2*np.pi*fs2))
    if str(params.keys())[10:].find("'fs2'") == -1: #elif fs == 'none':
        R2 = params['R2']
        Q2 = params['Q2']
        n2 = params['n2']

    Rs = params['Rs']
    return Rs + (R/(1+R*Q*(w*1j)**n)) + (R2/(1+R2*Q2*(w*1j)**n2))


def cir_Randles_simplified_Fit(params, w):
    """
    Fit Function: Randles simplified -Rs-(Q-(RW)-)-
    Return the impedance of a Randles circuit. See more under cir_Randles_simplified()

    NOTE: This Randles circuit is only meant for semi-infinate linear diffusion

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    """
    if str(params.keys())[10:].find("'R'") == -1: #if R == 'none':
        Q = params['Q']
        n = params['n']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("'Q'") == -1: #elif Q == 'none':
        R = params['R']
        n = params['n']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("'n'") == -1: #elif n == 'none':
        R = params['R']
        Q = params['Q']
        fs = params['fs']
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("'fs'") == -1: #elif fs == 'none':
        R = params['R']
        Q = params['Q']
        n = params['n']

    Rs = params['Rs']
    sigma = params['sigma']

    Z_Q = 1/(Q*(w*1j)**n)
    Z_R = R
    Z_w = sigma*(w**(-0.5))-1j*sigma*(w**(-0.5))

    return Rs + 1/(1/Z_Q + 1/(Z_R+Z_w))


def cir_RsRQQ_fit(params, w):
    """
    Fit Function: -Rs-RQ-Q-

    See cir_RsRQQ() for details
    """
    Rs = params['Rs']
    Q = params['Q']
    n = params['n']
    Z_Q = 1/(Q*(w*1j)**n)

    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    if str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ = (R1/(1+R1*Q1*(w*1j)**n1))

    return Rs + Z_RQ + Z_Q


def cir_RsRQC_fit(params, w):
    """
    Fit Function: -Rs-RQ-C-

    See cir_RsRQC() for details
    """
    Rs = params['Rs']
    C = params['C']
    Z_C = 1/(C*(w*1j))

    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    if str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ = (R1/(1+R1*Q1*(w*1j)**n1))

    return Rs + Z_RQ + Z_C


def cir_RsRCC_fit(params, w):
    """
    Fit Function: -Rs-RC-C-

    See cir_RsRCC() for details
    """
    Rs = params['Rs']
    R1 = params['R1']
    C1 = params['C1']
    C = params['C']
    return Rs + cir_RC(w, C=C1, R=R1, fs='none') + elem_C(w, C=C)


def cir_RsRCQ_fit(params, w):
    """
    Fit Function: -Rs-RC-Q-

    See cir_RsRCQ() for details
    """
    Rs = params['Rs']
    R1 = params['R1']
    C1 = params['C1']
    Q = params['Q']
    n = params['n']
    return Rs + cir_RC(w, C=C1, R=R1, fs='none') + elem_Q(w,Q,n)


def cir_C_RC_C_fit(params, w):
    """
    Fit Function: -C-(RC)-C-

    See cir_C_RC_C() for details

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    """
    # Interfacial impedance
    Ce = params['Ce']
    Z_C = 1/(Ce*(w*1j))

    # Bulk impendance
    if str(params.keys())[10:].find("Rb") == -1: #if R == 'none':
        Cb = params['Cb']
        fsb = params['fsb']
        Rb = (1/(Cb*(2*np.pi*fsb)))
    if str(params.keys())[10:].find("Cb") == -1: #elif Q == 'none':
        Rb = params['Rb']
        fsb = params['fsb']
        Cb = (1/(Rb*(2*np.pi*fsb)))
    if str(params.keys())[10:].find("fsb") == -1: #elif fs == 'none':
        Rb = params['Rb']
        Cb = params['Cb']
    Z_RC = (Rb/(1+Rb*Cb*(w*1j)))


    return Z_C + Z_RC


def cir_Q_RQ_Q_Fit(params, w):
    """
    Fit Function: -Q-(RQ)-Q-

    See cir_Q_RQ_Q() for details

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    """
    # Interfacial impedance
    Qe = params['Qe']
    ne = params['ne']
    Z_Q = 1/(Qe*(w*1j)**ne)

    # Bulk impedance
    if str(params.keys())[10:].find("Rb") == -1: #if R == 'none':
        Qb = params['Qb']
        nb = params['nb']
        fsb = params['fsb']
        Rb = (1/(Qb*(2*np.pi*fsb)**nb))
    if str(params.keys())[10:].find("Qb") == -1: #elif Q == 'none':
        Rb = params['Rb']
        nb = params['nb']
        fsb = params['fsb']
        Qb = (1/(Rb*(2*np.pi*fsb)**nb))
    if str(params.keys())[10:].find("nb") == -1: #elif n == 'none':
        Rb = params['Rb']
        Qb = params['Qb']
        fsb = params['fsb']
        nb = np.log(Qb*Rb)/np.log(1/(2*np.pi*fsb))
    if str(params.keys())[10:].find("fsb") == -1: #elif fs == 'none':
        Rb = params['Rb']
        nb = params['nb']
        Qb = params['Qb']
    Z_RQ =  Rb/(1+Rb*Qb*(w*1j)**nb)

    return Z_Q + Z_RQ


def cir_RCRCZD_fit(params, w):
    """
    Fit Function: -RC_b-RC_e-Z_D

    See cir_RCRCZD() for details

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    """
    # Interfacial impendace
    if str(params.keys())[10:].find("Re") == -1: #if R == 'none':
        Ce = params['Ce']
        fse = params['fse']
        Re = (1/(Ce*(2*np.pi*fse)))
    if str(params.keys())[10:].find("Ce") == -1: #elif Q == 'none':
        Re = params['Rb']
        fse = params['fsb']
        Ce = (1/(Re*(2*np.pi*fse)))
    if str(params.keys())[10:].find("fse") == -1: #elif fs == 'none':
        Re = params['Re']
        Ce = params['Ce']
    Z_RCe = (Re/(1+Re*Ce*(w*1j)))

    # Bulk impendance
    if str(params.keys())[10:].find("Rb") == -1: #if R == 'none':
        Cb = params['Cb']
        fsb = params['fsb']
        Rb = (1/(Cb*(2*np.pi*fsb)))
    if str(params.keys())[10:].find("Cb") == -1: #elif Q == 'none':
        Rb = params['Rb']
        fsb = params['fsb']
        Cb = (1/(Rb*(2*np.pi*fsb)))
    if str(params.keys())[10:].find("fsb") == -1: #elif fs == 'none':
        Rb = params['Rb']
        Cb = params['Cb']
    Z_RCb = (Rb/(1+Rb*Cb*(w*1j)))

    # Mass transport impendance
    L = params['L']
    D_s = params['D_s']
    u1 = params['u1']
    u2 = params['u2']

    alpha = ((w*1j*L**2)/D_s)**(1/2)
    Z_D = Rb * (u2/u1) * (tanh(alpha)/alpha)
    return Z_RCb + Z_RCe + Z_D


def cir_RsTLsQ_fit(params, w):
    """
    Fit Function: -Rs-TLsQ-
    TLs = Simplified Transmission Line, with a non-faradaic interfacial impedance (Q)
    See more under cir_RsTLsQ()

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    """
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    Q = params['Q']
    n = params['n']

    Phi = 1/(Q*(w*1j)**n)
    X1 = Ri # ohm/cm
    Lam = (Phi/X1)**(1/2) #np.sqrt(Phi/X1)

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)  #Handles coth with x having very large or very small numbers
#
#    Z_TLsQ = Lam * X1 * coth_mp
    Z_TLsQ = Lam * X1 * coth(x)

    return Rs + Z_TLsQ


def cir_RsRQTLsQ_Fit(params, w):
    """
    Fit Function: -Rs-RQ-TLsQ-
    TLs = Simplified Transmission Line, with a non-faradaic interfacial impedance (Q)
    See more under cir_RsRQTLsQ

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    """
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    Q = params['Q']
    n = params['n']

    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    if str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ = (R1/(1+R1*Q1*(w*1j)**n1))


    Phi = 1/(Q*(w*1j)**n)
    X1 = Ri
    Lam = (Phi/X1)**(1/2)

    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers

    Z_TLsQ = Lam * X1 * coth_mp

    return Rs + Z_RQ + Z_TLsQ


def cir_RsTLs_Fit(params, w):
    """
    Fit Function: -Rs-RQ-TLs-
    TLs = Simplified Transmission Line, with a faradaic interfacial impedance (RQ)
    See mor under cir_RsTLs()

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    """
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']

    if str(params.keys())[10:].find("R") == -1: #if R == 'none':
        Q = params['Q']
        n = params['n']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("Q") == -1: #elif Q == 'none':
        R = params['R']
        n = params['n']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("n") == -1: #elif n == 'none':
        R = params['R']
        Q = params['Q']
        fs = params['fs']
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("fs") == -1: #elif fs == 'none':
        R = params['R']
        n = params['n']
        Q = params['Q']
    Phi = R/(1+R*Q*(w*1j)**n)

    X1 = Ri
    Lam = (Phi/X1)**(1/2)
    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers

    Z_TLs = Lam * X1 * coth_mp

    return Rs + Z_TLs


def cir_RsRQTLs_Fit(params, w):
    """
    Fit Function: -Rs-RQ-TLs-
    TLs = Simplified Transmission Line with a faradaic interfacial impedance (RQ)
    See more under cir_RsRQTLs()

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    """
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']

    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    if str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ = (R1/(1+R1*Q1*(w*1j)**n1))

    if str(params.keys())[10:].find("R2") == -1: #if R == 'none':
        Q2 = params['Q2']
        n2 = params['n2']
        fs2 = params['fs2']
        R2 = (1/(Q2*(2*np.pi*fs2)**n2))
    if str(params.keys())[10:].find("Q2") == -1: #elif Q == 'none':
        R2 = params['R2']
        n2 = params['n2']
        fs2 = params['fs2']
        Q2 = (1/(R2*(2*np.pi*fs2)**n1))
    if str(params.keys())[10:].find("n2") == -1: #elif n == 'none':
        R2 = params['R2']
        Q2 = params['Q2']
        fs2 = params['fs2']
        n2 = np.log(Q2*R2)/np.log(1/(2*np.pi*fs2))
    if str(params.keys())[10:].find("fs2") == -1: #elif fs == 'none':
        R2 = params['R2']
        n2 = params['n2']
        Q2 = params['Q2']
    Phi = (R2/(1+R2*Q2*(w*1j)**n2))
    X1 = Ri
    Lam = (Phi/X1)**(1/2)

    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers

    Z_TLs = Lam * X1 * coth_mp

    return Rs + Z_RQ + Z_TLs


def cir_RsTLQ_fit(params, w):
    """
    Fit Function: -R-TLQ- (interface non-reacting, i.e. blocking electrode)
    Transmission line w/ full complexity, which both includes Ri and Rel

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    """
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    Rel = params['Rel']
    Q = params['Q']
    n = params['n']

    #The impedance of the series resistance
    Z_Rs = Rs

    # The Interfacial impedance is given by an -(RQ)- circuit
    Phi = elem_Q(w, Q=Q, n=n)
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_TL


def cir_RsRQTLQ_fit(params, w):
    """
    Fit Function: -R-RQ-TLQ- (interface non-reacting, i.e. blocking electrode)
    Transmission line w/ full complexity, which both includes Ri and Rel

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    """
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    Rel = params['Rel']
    Q = params['Q']
    n = params['n']

    #The impedance of the series resistance
    Z_Rs = Rs

    #The (RQ) circuit in series with the transmission line
    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    if str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ1 = (R1/(1+R1*Q1*(w*1j)**n1))

    # The Interfacial impedance is given by an -(RQ)- circuit
    Phi = elem_Q(w, Q=Q, n=n)
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_RQ1 + Z_TL


def cir_RsTL_Fit(params, w):
    """
    Fit Function: -R-TLQ- (interface reacting, i.e. non-blocking)
    Transmission line w/ full complexity, which both includes Ri and Rel

    See cir_RsTL() for details

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    """
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    Rel = params['Rel']

    #The impedance of the series resistance
    Z_Rs = Rs

    # The Interfacial impedance is given by an -(RQ)- circuit
    if str(params.keys())[10:].find("R") == -1: #if R == 'none':
        Q = params['Q']
        n = params['n']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("Q") == -1: #elif Q == 'none':
        R = params['R']
        n = params['n']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("n") == -1: #elif n == 'none':
        R = params['R']
        Q = params['Q']
        fs = params['fs']
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("fs") == -1: #elif fs == 'none':
        R = params['R']
        n = params['n']
        Q = params['Q']

    Phi = (R/(1+R*Q*(w*1j)**n))
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_TL


def cir_RsRQTL_fit(params, w):
    """
    Fit Function: -R-RQ-TL- (interface reacting, i.e. non-blocking)
    Transmission line w/ full complexity including both includes Ri and Rel

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    """
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    Rel = params['Rel']

    #The impedance of the series resistance
    Z_Rs = Rs

    # The Interfacial impedance is given by an -(RQ)- circuit
    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    elif str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    elif str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    elif str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ1 = (R1/(1+R1*Q1*(w*1j)**n1))
#
#    # The Interfacial impedance is given by an -(RQ)- circuit
    if str(params.keys())[10:].find("R2") == -1: #if R == 'none':
        Q2 = params['Q2']
        n2 = params['n2']
        fs2 = params['fs2']
        R2 = (1/(Q2*(2*np.pi*fs2)**n2))
    elif str(params.keys())[10:].find("Q2") == -1: #elif Q == 'none':
        R2 = params['R2']
        n2 = params['n2']
        fs2 = params['fs2']
        Q2 = (1/(R2*(2*np.pi*fs2)**n1))
    elif str(params.keys())[10:].find("n2") == -1: #elif n == 'none':
        R2 = params['R2']
        Q2 = params['Q2']
        fs2 = params['fs2']
        n2 = np.log(Q2*R2)/np.log(1/(2*np.pi*fs2))
    elif str(params.keys())[10:].find("fs2") == -1: #elif fs == 'none':
        R2 = params['R2']
        n2 = params['n2']
        Q2 = params['Q2']
    Phi = (R2/(1+R2*Q2*(w*1j)**n2))

    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float((mp.coth(x_mp[i]).imag))*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(((1-mp.exp(-2*x_mp[i]))/(2*mp.exp(-x_mp[i]))).real) + float(((1-mp.exp(-2*x_mp[i]))/(2*mp.exp(-x_mp[i]))).real)*1j)
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float((mp.sinh(x_mp[i]).imag))*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_RQ1 + Z_TL


def cir_RsTL_1Dsolid_fit(params, w):
    """
    Fit Function: -R-TL(Q(RW))-
    Transmission line w/ full complexity

    See cir_RsTL_1Dsolid() for details

    David Brown (demoryb@berkeley.edu)
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    """
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    radius = params['radius']
    D = params['D']
    R = params['R']
    Q = params['Q']
    n = params['n']
    R_w = params['R_w']
    n_w = params['n_w']
    Rel = params['Rel']
    Ri = params['Ri']

    #The impedance of the series resistance
    Z_Rs = Rs

    #The impedance of a 1D Warburg Element
    time_const = (radius**2)/D

    x = (time_const*w*1j)**n_w
    x_mp = mp.matrix(x)
    warburg_coth_mp = []
    for i in range(len(w)):
        warburg_coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)

    Z_w = R_w * np.array(warburg_coth_mp)/x

    # The Interfacial impedance is given by a Randles Equivalent circuit with the finite space warburg element in series with R2
    Z_Rct = R
    Z_Q = elem_Q(w=w, Q=Q, n=n)
    Z_Randles = 1/(1/Z_Q + 1/(Z_Rct+Z_w)) #Ohm

    # The Impedance of the Transmission Line
    lamb = (Z_Randles/(Rel+Ri))**(1/2)
    x = L/lamb
#    lamb_mp = mp.matrix(x)
#    sinh_mp = []
#    coth_mp = []
#    for j in range(len(lamb_mp)):
#        sinh_mp.append(float(mp.sinh(lamb_mp[j]).real)+float((mp.sinh(lamb_mp[j]).imag))*1j)
#        coth_mp.append(float(mp.coth(lamb_mp[j]).real)+float(mp.coth(lamb_mp[j]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/np.array(sinh_mp))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/sinh(x))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_TL


def cir_RsRQTL_1Dsolid_fit(params, w):
    """
    Fit Function: -R-RQ-TL(Q(RW))-
    Transmission line w/ full complexity, which both includes Ri and Rel. The Warburg element is specific for 1D solid-state diffusion

    See cir_RsRQTL_1Dsolid() for details

    David Brown (demoryb@berkeley.edu)
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    """
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    radius = params['radius']
    D = params['D']
    R2 = params['R2']
    Q2 = params['Q2']
    n2 = params['n2']
    R_w = params['R_w']
    n_w = params['n_w']
    Rel = params['Rel']
    Ri = params['Ri']
    #The impedance of the series resistance
    Z_Rs = Rs

    # The Interfacial impedance is given by an -(RQ)- circuit
    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    elif str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    elif str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    elif str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ1 = (R1/(1+R1*Q1*(w*1j)**n1))

    #The impedance of a 1D Warburg Element
    time_const = (radius**2)/D

    x = (time_const*w*1j)**n_w
    x_mp = mp.matrix(x)
    warburg_coth_mp = []
    for i in range(len(w)):
        warburg_coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)

    Z_w = R_w * np.array(warburg_coth_mp)/x

    # The Interfacial impedance is given by a Randles Equivalent circuit with the finite space warburg element in series with R2
    Z_Rct = R2
    Z_Q = elem_Q(w,Q=Q2,n=n2)
    Z_Randles = 1/(1/Z_Q + 1/(Z_Rct+Z_w)) #Ohm

    # The Impedance of the Transmission Line
    lamb = (Z_Randles/(Rel+Ri))**(1/2)
    x = L/lamb
#    lamb_mp = mp.matrix(x)
#    sinh_mp = []
#    coth_mp = []
#    for j in range(len(lamb_mp)):
#        sinh_mp.append(float(mp.sinh(lamb_mp[j]).real)+float((mp.sinh(lamb_mp[j]).imag))*1j)
#        coth_mp.append(float(mp.coth(lamb_mp[j]).real)+float(mp.coth(lamb_mp[j]).imag)*1j)
#
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/np.array(sinh_mp))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/sinh(x))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_RQ1 + Z_TL

# For the outside world
CIRCUIT_DICT = {
    'C': elem_C_fit,
    'Q': elem_Q_fit,
    'R-C': cir_RsC_fit,
    'R-Q': cir_RsQ_fit,
    'RC': cir_RC_fit,
    'RQ': cir_RQ_fit,
    'R-RQ': cir_RsRQ_fit,
    'R-RQ-RQ': cir_RsRQRQ_fit,
    'L-R-RQ-RQ': cir_LRsRQRQ_fit,
    'L-R-RQ-RQ-RQ': cir_LRsRQRQRQ_fit,
    'L-R-Q(R)-Q(R-Q(R))-Q(R)': cir_LRQRQRQRQR_fit,
    'R-RC-C': cir_RsRCC_fit,
    'R-RC-Q': cir_RsRCQ_fit,
    'R-RQ-Q': cir_RsRQQ_fit,
    'R-RQ-C': cir_RsRQC_fit,
    'R-(Q(RW))': cir_Randles_simplified_Fit,
    # 'R-(Q(RM))': cir_Randles_uelectrode_fit,
    'C-RC-C': cir_C_RC_C_fit,
    'Q-RQ-Q': cir_Q_RQ_Q_Fit,
    'RC-RC-ZD': cir_RCRCZD_fit,
    'R-TLsQ': cir_RsRQTLsQ_Fit,
    'R-RQ-TLsQ': cir_RsRQTLsQ_Fit,
    'R-TLs': cir_RsTLs_Fit,
    'R-RQ-TLs': cir_RsRQTLs_Fit,
    'R-TLQ': cir_RsTLQ_fit,
    'R-RQ-TLQ': cir_RsRQTLQ_fit,
    'R-Tl': cir_RsTL_Fit,
    'R-RQ-TL': cir_RsRQTL_fit,
    'R-TL1Dsolid': cir_RsTL_1Dsolid_fit,
    'R-RQ-TL1Dsolid': cir_RsRQTL_1Dsolid_fit,
}


def leastsq_errorfunc(params, w, re, im, circuit, weight_func):
    """
    Sum of squares error function for the complex non-linear least-squares fitting procedure (CNLS).
    The fitting function (lmfit) will use this function to iterate over until the total sum of
    errors is minimized.

    During the minimization the fit is weighed, and currently three different weigh options are avaliable:
        - modulus
        - unity
        - proportional

    Modulus is generially recommended as random errors and a bias can exist in the experimental data.

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ------------
    - params: parameters needed for CNLS
    - re: real impedance
    - im: Imaginary impedance
    - circuit:
      The avaliable circuits are the keys of CIRCUIT_DICT
    - weight_func
      Weight function
        - modulus
        - unity
        - proportional
    """
    if circuit in list(CIRCUIT_DICT.keys()):
        re_fit = CIRCUIT_DICT[circuit](params, w).real
        im_fit = -CIRCUIT_DICT[circuit](params, w).imag
    else:
        raise ValueError(f'circuit {circuit} is not one of the supported circuit strings')

    # sum of squares
    error = [(re-re_fit)**2, (im-im_fit)**2]

    # Different Weighing options, see Lasia
    if weight_func == 'modulus':
        weight = [1/((re_fit**2 + im_fit**2)**(1/2)), 1/((re_fit**2 + im_fit**2)**(1/2))]
    elif weight_func == 'proportional':
        weight = [1/(re_fit**2), 1/(im_fit**2)]
    elif weight_func == 'unity':
        unity_1s = []
        for k in range(len(re)):
            # makes an array of [1]'s, so that the weighing is == 1 * sum of squres.
            unity_1s.append(1)
        weight = [unity_1s, unity_1s]
    else:
        raise ValueError('weight not defined in leastsq_errorfunc()')

    # weighted sum of squares
    S = np.array(weight) * error
    return S
