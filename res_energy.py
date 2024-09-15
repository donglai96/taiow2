

from signal import raise_signal
import numpy as np
import constants as cst
import sympy as sym
from taiphysics import *



def get_resonance_velocity(w, kpara, alpha, wce, n):
    v = sym.Symbol('v')

    lhs = w - kpara * v * np.cos(alpha)

    rhs = n * -1 * wce*(1 - (v**2)/cst.C**2)**0.5


    # print('lhs is ', lhs)
    # print('rhs is ', rhs)
    print(" resonance n is ", n )
    # print("the solution of v  are")
    #print(sym.solve(lhs - rhs, v))
    return sym.solve(lhs - rhs, v)


def get_resonance_whistler(w, wpe,wce,  alpha, nres,psi):
    """Calculate the resonance velocity


    Args:
        w (_type_):frequency of wave
        wpe (_type_): plasma frequency
        wce (_type_): gyro frequency , positive
        kpara (_type_): k in parallel direction with background magnetic field
        alpha (_type_): pitchangle of the particle
        nres (_type_): resonance number
        psi (_type_): wave normal angle
    """
    ws = w
    RR = 1 + wpe**2 / ((wce - ws) * ws) 
    LL = 1 - wpe**2 / ((wce + ws) * ws)
    PP = 1 - wpe**2 / (ws * ws)
    SS = 1/2 * (RR + LL)
    DD = 1/2 * (RR - LL)
        
        # calculate the k value
    sin = np.sin(psi)
    cos = np.cos(psi)
    rhs_up = RR * LL * sin**2 + PP * SS * (1 + cos**2) - ((RR * LL - PP *SS)**2 * sin**4 + 4 * PP**2 * DD**2 * cos**2)**0.5
    rhs_bot = 2 * (SS * sin**2 + PP*cos**2)
    k = (rhs_up / rhs_bot) ** 0.5 * ws /cst.C 

    kpara = k*np.cos(psi)

    v_result = get_resonance_velocity(w, kpara, alpha, wce, nres)
    v = v_result[0]
    if v< 0 :
        print('Check the pitch angle!')
        print('Changing the initial solution')
        #raise Warning("The velocity is negative, use")
        v = v_result[1]
    if np.abs(v) > cst.C:
        raise ValueError('The velocity is larger than light speed')
    print("The resonating velocity is ",v/cst.C,"c")
    gamma = 1 / (1 - (v**2)/cst.C**2)**0.5
    p = gamma * v * cst.Me
    return p, k