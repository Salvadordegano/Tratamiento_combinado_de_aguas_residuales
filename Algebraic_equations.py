import math
import numpy as np
from scipy.optimize import fsolve

# Inhibition terms
def inhibition1(ki, s):
    inh1 = ki/(ki+s)
    return inh1
    
    
def inhibitionpH(pH_ll, pH_ul, pH):
    inpH = (1+2*10**(0.5*(pH_ll-pH_ul)))/(1+10**(pH-pH_ul)+10**(pH_ll-pH))
    return inpH

def inhibitionpH2(pH_ll, pH_ul, pH):
    if pH > pH_ul:
        I_ph = 1
    else:
        I_ph = math.exp(-3*((pH-pH_ul)/(pH_ul-pH_ll))**2)
    return I_ph


def inhibition2(ks, s): #nitrógeno inorgánico
    inh2 = s/(s+ks)
    return inh2


# Reaction rates
def hydrolysis(k_hyd, s):
    hydr_rate = k_hyd*s
    return hydr_rate


def rates(km, s, ks):
    rate = km * (s/(ks+s))
    return rate
    

def lcfa_rate(km, ks, s, ki_lcfa):
    rate = km * (s/(1+(ks/s)+(s/ki_lcfa)))
    return rate

    
def biomass_dead(k_dead, biomass):
    dead_rate = k_dead * biomass
    return dead_rate


# Mass transfer terms
def kH_temp(KH_298, deltaH_KH, T):
    kH_corregido = KH_298*math.exp((deltaH_KH/8.324)*((1/298.15)-(1/T)))
    return kH_corregido


def rho_gas(lamb, kLa, s, kH, p):
    rho = lamb*kLa*(s-kH*p)
    return rho


# Parcial preassures
def pres(gas, T):
    pp = gas*0.08314472*T
    return pp


def p_tot(p1, p2, p3):
    pres = p1 + p2 + p3
    return pres
    
    
# gas flow
def gas_flow(V_liq, rhoch4, rhoco2, ptot, p_h2o, T):
    q_gas = ((0.08314472*T)/(ptot-p_h2o))*V_liq*((rhoch4/64)+rhoco2)
    return q_gas

def gas_flow2(V_liq, rhoch4, rhoco2, rhoh2, ptot, p_h2o, T):
    q_gas = ((0.08314472*T)/(ptot-p_h2o))*V_liq*((rhoch4/64)+(rhoh2/16)+rhoco2)
    return q_gas

def methane_flow(V_liq, rhoch4, ptot, p_h2o, T):
    q_methane = ((0.08314472*T)/(ptot-p_h2o))*V_liq*(rhoch4/64)
    return q_methane
       
def pH(cat, i_n, ka_nh3, i_c, ka_co2, lcfa, ka_lcfa, total_ac, ka_ac, total_pro, ka_pro, total_bu, ka_bu, total_val, ka_val, ka_w, an):
    def charge_balance(y):
            return cat + ((i_n*y)/(ka_nh3+y)) + y - (i_c - ((i_c*y)/(ka_co2+y))) - ((lcfa - ((lcfa*y)/(ka_lcfa+y)))/736) - ((total_ac - ((total_ac*y)/(ka_ac+y)))/64) - ((total_pro - ((total_pro*y)/(ka_pro+y)))/112) - ((total_bu - ((total_bu*y)/(ka_bu+y)))/160) - ((total_val - ((total_val*y)/(ka_val+y)))/208) - (ka_w/y) - an        

    s_h = float(fsolve(charge_balance, 0.0000001))
    p_H = -1*math.log10(s_h)
    return p_H

def pH1(s_h):
    p_H = -1*math.log10(s_h)
    return p_H

# Acid-base equilibrium    
def acid_base(s, ka, s_h):
    cation = (s_h*s)/(ka+s_h)
    anion = s - cation
    return anion

# Temperature-related growth
def sigma(shg):
    sig = (-((shg**2)/(2*math.log(0.5))))**(0.5)
    return sig

# Measured variables
def FOS(ac, pro, but, val):
    agv = ((ac/60 + pro/74 + but/88 + val/102)*60)*1000
    return agv

def TAC(hco3, ac, pro, but, val):
    ta = ((hco3 + ac/60 + pro/74 + but/88 + val/102)*100)*1000
    return ta

def FOS_TAC(fos,tac):
    agv_ta = fos/tac
    return agv_ta

def N_NH3(s_in):
    nat = s_in * 14 * 1000
    return nat

def net_energy(r_e, q_methane, rend_CHP, l_bio_CHP):
    e_neta = (1-r_e)*(q_methane*35.8)*rend_CHP*(1-l_bio_CHP)/3600
    return e_neta
