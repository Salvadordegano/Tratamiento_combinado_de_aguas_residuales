import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import Algebraic_equations as ae

# The Tanque_EfluenteAnaerobio class implement mass balance equation without reaction terms

class Tanque_EfluenteAnaerobio():
    def __init__(self, cts, feed, initial, tspan, teval):
        self.cts = cts
        self.feed = feed
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
        # Concentration of lipolysis degraders (gCOD/L)
        x_lip = x[15]
        # Concentration of glucose degraders (gCOD/L)
        x_gl = x[16]
        # Concentration of aminoacids degraders (gCOD/L)
        x_aa = x[17]
        # Concentration of lcfa degraders (gCOD/L) 
        x_lcfa = x[18]
        # Concentration of valerate degraders (gCOD/L)
        x_va = x[19]
        # Concentration of butirate degraders (gCOD/L)
        x_bu = x[20]
        # Concentration of propionate degraders (gCOD/L)
        x_pro = x[21]
        # Concentration of acetate degraders (gCOD/L)
        x_ac = x[22]
        

        #  feed = Feed Concentration (23)
        # inlet of inerts carbohydrates (gCOD/L)
        ch_i_in = self.feed[0]
        # inlet of inerts proteins (gCOD/L)
        pro_i_in = self.feed[1]
        # inlet of carbohydrates (gCOD/L)
        ch_in = self.feed[2]
        # inlet of proteins (gCOD/L)
        pro_in = self.feed[3]
        # inlet of lipids (gCOD/L)
        lip_in = self.feed[4]
        # inlet of glucose (gCOD/L)
        gl_in = self.feed[5]
        # inlet of aminoacids (gCOD/L)
        aa_in = self.feed[6]
        # inlet of Long Chain Fatty Acids (gCOD/L)
        lcfa_in = self.feed[7]
        # inlet of total valerate (gCOD/L)
        va_tot_in = self.feed[8]
        # inlet of total butirate (gCOD/L)
        bu_tot_in = self.feed[9]
        # inlet of total propionate (gCOD/L)
        pro_tot_in = self.feed[10]
        # inlet of total acetate (gCOD/L) 
        ac_tot_in = self.feed[11]
        # inlet of inorganic carbon (mol/L)
        i_c_in = self.feed[12]
        # inlet of inorganic nitrogen (mol/L)
        i_n_in = self.feed[13]
        # lipolysis degraders (gCOD/L)
        x_lip_in = self.feed[14]
        # glucose degraders (gCOD/L)
        x_gl_in = self.feed[15]
        # aminoacids degraders (gCOD/L)
        x_aa_in = self.feed[16]
        # lcfa degraders (gCOD/L)
        x_lcfa_in = self.feed[17]
        # valerate degraders (gCOD/L)
        x_va_in = self.feed[18]
        # butyrate degraders (gCOD/L)
        x_bu_in = self.feed[19]
        # propionate degraders (gCOD/L)
        x_pro_in = self.feed[20]
        # acetate degraders (gCOD/L)
        x_ac_in = self.feed[21]

        #  cts   = Constants (36)
        #   Flows
        #    inlet flow
        qin = self.cts[0]
        #    outlet flow
        qout = self.cts[1]


        # Mass balance: volume derivative
        dV_liq = qin - qout
        # Species balance: concentration derivative
        # Chain rule: d(V*Ca)/dt = Ca * dV/dt + V * dCa/dt
        # Inerts protein mass balance
        dpro_idt = (qin*pro_i_in- qout*pro_i)/V_liq - (pro_i*dV_liq/V_liq)
        # Inerts carbohydrates mass balance
        dch_idt = (qin*ch_i_in- qout*ch_i)/V_liq - (ch_i*dV_liq/V_liq)
        # carbohydrates mass balance
        dchdt = (qin*ch_in- qout*ch)/V_liq - (ch*dV_liq/V_liq)
        # protein mass balance
        dprotdt = (qin*pro_in- qout*prot)/V_liq - (prot*dV_liq/V_liq)
        # lipids mass balance
        dlipdt = (qin*lip_in- qout*lip)/V_liq - (lip*dV_liq/V_liq)
        # inorganic carbon mass balance
        di_cdt = (qin*i_c_in- qout*i_c)/V_liq - (i_c*dV_liq/V_liq)
        # inorganic nitrogen mass balance
        di_ndt = (qin*i_n_in- qout*i_n)/V_liq - (i_n*dV_liq/V_liq)
        # Inerts protein mass balance
        dgldt = (qin*gl_in- qout*gl)/V_liq - (gl*dV_liq/V_liq)
        # Inerts carbohydrates mass balance
        daadt = (qin*aa_in- qout*aa)/V_liq - (aa*dV_liq/V_liq)
        # carbohydrates mass balance
        dlcfadt = (qin*lcfa_in - qout*lcfa)/V_liq - (lcfa*dV_liq/V_liq)
        # protein mass balance
        dva_totdt = (qin*va_tot_in - qout*va_total)/V_liq - (va_total*dV_liq/V_liq)
        # lipids mass balance
        dbu_totdt = (qin*bu_tot_in - qout*bu_total)/V_liq - (bu_total*dV_liq/V_liq)
        # inorganic carbon mass balance
        dpro_totdt = (qin*pro_tot_in - qout*pro_total)/V_liq - (pro_total*dV_liq/V_liq)
        # inorganic nitrogen mass balance
        dac_totdt = (qin*ac_tot_in - qout*ac_total)/V_liq - (ac_total*dV_liq/V_liq)
        # lipids degraders mass balance
        dx_lipdt = (qin*x_lip_in- qout*x_lip)/V_liq - (x_lip*dV_liq/V_liq)
        # glucose degraders mass balance
        dx_gldt = (qin*x_gl_in - qout*x_gl)/V_liq - (x_gl*dV_liq/V_liq)
        # aminoacids degraders mass balance
        dx_aadt = (qin*x_aa_in - qout*x_aa)/V_liq - (x_aa*dV_liq/V_liq)
        # lcfa degraders mass balance
        dx_lcfadt = (qin*x_lcfa_in - qout*x_lcfa)/V_liq - (x_lcfa*dV_liq/V_liq)
        # valerate degraders mass balance
        dx_vadt = (qin*x_va_in - qout*x_va)/V_liq - (x_va*dV_liq/V_liq)
        # butyrate degraders mass balance
        dx_budt = (qin*x_bu_in - qout*x_bu)/V_liq - (x_bu*dV_liq/V_liq)
        # propionic degraders mass balance
        dx_prodt = (qin*x_pro_in - qout*x_pro)/V_liq - (x_pro*dV_liq/V_liq)
        # acetate degraders mass balance
        dx_acdt = (qin*x_ac_in - qout*x_ac)/V_liq - (x_ac*dV_liq/V_liq)


        f = np.array([dV_liq, dch_idt, dpro_idt, dchdt, dprotdt, dlipdt, di_cdt, di_ndt, dgldt, daadt, dlcfadt, dva_totdt, dbu_totdt, dpro_totdt, dac_totdt, dx_lipdt, dx_gldt, dx_aadt, dx_lcfadt, dx_vadt, dx_budt, dx_prodt, dx_acdt])
        # Return derivatives
        return f


    # Function that solves the DAE system
    def ode(self):
        
        # Differential equations
        states = solve_ivp(self.vessel, self.tspan, self.initial, t_eval=self.teval, method="BDF", rtol=1E-9, atol=1E-12)
        
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
        biomass_lip = states.y[15]
        biomass_su = states.y[16]
        biomass_aa = states.y[17]
        biomass_lcfa = states.y[18]
        biomass_va = states.y[19]
        biomass_bu = states.y[20]
        biomass_pro = states.y[21]
        biomass_ac = states.y[22]

        S_Sout = sugars + acetate + propionate + valerate + butyrate + aminoacids + lcfa

        X_Sout = lipids + proteins + carbohydrates + ch_inert + prot_inert + biomass_lip + biomass_bu + biomass_lcfa + biomass_pro + biomass_ac + biomass_va + biomass_su + biomass_aa

        S_Iout = ch_inert + prot_inert

        S_NH4out = inorganic_nitrogen*14

        S_Alkout = inorganic_carbon

        return time, v, ch_inert, prot_inert, carbohydrates, proteins, lipids, inorganic_carbon, inorganic_nitrogen, sugars, aminoacids, lcfa, valerate, butyrate, propionate, acetate, biomass_lip, biomass_su, biomass_aa, biomass_lcfa, biomass_va, biomass_bu, biomass_pro, biomass_ac, S_Sout, S_Iout, X_Sout, S_NH4out, S_Alkout
