import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import Algebraic_equations as ae

# The Tanque_Recepcion class implement mass balance equation without reaction terms

class Tanque_Sustrato():
    def __init__(self, cts, initial, tspan, teval):
        self.cts = cts
        self.initial = initial
        self.tspan = tspan
        self.teval = teval


    #CSTR's FUNCTION - Differential-Algebraic Equation System (DAE)
    def vessel(self, t, x):
        # States (26):
        # Liquid volume (L)
        V_liq = x[0]
        # Concentration of inerts carbohydrates (gCOD/L)
        ch_i = x[1]
        # Concentration of inerts proteins (gCOD/L)
        pro_i = x[2]
        # Concentration of carbohydrates (gCOD/L)
        ch = x[3]
        # Concentration of proteins (gCOD/L)
        prot = x[4]
        # Concentration of lipids (gCOD/L)
        lip = x[5]
        # Concentration of inorganic carbon (mol/L)
        i_c = x[6]
        # Concentration of inorganic nitrogen (mol/L)
        i_n = x[7]
        # Concentration of glucose (g/L)
        gl = x[8]
        # Concentration of aminoacids (g/L)
        aa = x[9]
        # Concentration of LCFA (g/L)
        lcfa = x[10]
        # Concentration of Tota Valerate (g/L)
        va_total = x[11]
        # Concentration of Tota Butyrate (g/L)
        bu_total = x[12]
        # Concentration of Tota Propionate (g/L)
        pro_total = x[13]
        # Concentration of Tota Acetate (g/L)
        ac_total = x[14]
        # Concentration of Cations [mol/L]
        cat = x[15]
        # Concentration of Anions [mol/L]
        an = x[16]

        #  cts   = Constants (36)
        #   Flows
        #    inlet flow
        qby = self.cts[0]
        #    outlet flow
        qout = self.cts[1]


        # Mass balance: volume derivative
        dV_liq = -qby - qout
        # Species balance: concentration derivative
        # Chain rule: d(V*Ca)/dt = Ca * dV/dt + V * dCa/dt
        # Inerts protein mass balance
        dpro_idt = (- (qout+qby)*pro_i)/V_liq - (pro_i*dV_liq/V_liq)
        # Inerts carbohydrates mass balance
        dch_idt = (- (qout+qby)*ch_i)/V_liq - (ch_i*dV_liq/V_liq)
        # carbohydrates mass balance
        dchdt = (- (qout+qby)*ch)/V_liq - (ch*dV_liq/V_liq)
        # protein mass balance
        dprotdt = (- (qout+qby)*prot)/V_liq - (prot*dV_liq/V_liq)
        # lipids mass balance
        dlipdt = (- (qout+qby)*lip)/V_liq - (lip*dV_liq/V_liq)
        # inorganic carbon mass balance
        di_cdt = (- (qout+qby)*i_c)/V_liq - (i_c*dV_liq/V_liq)
        # inorganic nitrogen mass balance
        di_ndt = (- (qout+qby)*i_n)/V_liq - (i_n*dV_liq/V_liq)
        # Inerts protein mass balance
        dgldt = (- (qout+qby)*gl)/V_liq - (gl*dV_liq/V_liq)
        # Inerts carbohydrates mass balance
        daadt = (- (qout+qby)*aa)/V_liq - (aa*dV_liq/V_liq)
        # carbohydrates mass balance
        dlcfadt = (- (qout+qby)*lcfa)/V_liq - (lcfa*dV_liq/V_liq)
        # protein mass balance
        dva_totdt = (- (qout+qby)*va_total)/V_liq - (va_total*dV_liq/V_liq)
        # lipids mass balance
        dbu_totdt = (- (qout+qby)*bu_total)/V_liq - (bu_total*dV_liq/V_liq)
        # inorganic carbon mass balance
        dpro_totdt = (- (qout+qby)*pro_total)/V_liq - (pro_total*dV_liq/V_liq)
        # inorganic nitrogen mass balance
        dac_totdt = (- (qout+qby)*ac_total)/V_liq - (ac_total*dV_liq/V_liq)
        # Cation mass balance
        dcatdt = (- (qout+qby)*cat)/V_liq - (cat*dV_liq/V_liq)
        # Cation mass balance
        dandt = (- (qout+qby)*an)/V_liq - (an*dV_liq/V_liq)

        f = np.array([dV_liq, dch_idt, dpro_idt, dchdt, dprotdt, dlipdt, di_cdt, di_ndt, dgldt, daadt, dlcfadt, dva_totdt, dbu_totdt, dpro_totdt, dac_totdt, dcatdt, dandt])
        # Return derivatives
        return f


    # Function that solves the DAE system
    def ode(self):
        
        # Differential equations
        states = solve_ivp(self.vessel, self.tspan, self.initial, t_eval=self.teval, method="BDF", rtol=1E-12, atol=1E-15)
        
        time = states.t
        v = states.y[0]
        ch_inert = states.y[1]
        prot_inert = states.y[2]
        carbohydrates = states.y[3]
        proteins = states.y[4]
        lipids = states.y[5]
        inorganic_carbon = states.y[6]
        inorganic_nitrogen = states.y[7]
        sugars = states.y[8]
        aminoacids = states.y[9]
        lcfa = states.y[10]
        valerate = states.y[11]
        butyrate = states.y[12]
        propionate = states.y[13]
        acetate = states.y[14]
        cations = states.y[15]
        anions = states.y[16]

        S_S = sugars + acetate + propionate + valerate + butyrate + aminoacids + lcfa

        S_Iout = ch_inert + prot_inert

        X_S = lipids + proteins + carbohydrates + ch_inert + prot_inert 

        i_NSS_Sus = (0.08289756*0.65)/S_S

        i_NXS_Sus = (0.08289756*(1-0.65))/X_S

        S_NH4by = inorganic_nitrogen*14

        return time, v, ch_inert, prot_inert, carbohydrates, proteins, lipids, inorganic_carbon, inorganic_nitrogen, sugars, aminoacids, lcfa, valerate, butyrate, propionate, acetate, S_S, S_Iout, X_S, i_NSS_Sus, i_NXS_Sus, S_NH4by, cations, anions
