import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, root
import Algebraic_equations as ae

# The Tanque_Digestor class implement the Angelidaki et al. (1999) model of Anaerobic Digestion

class Tanque_Aerobio():
    def __init__(self, pars, cts, feed, initial, tspan, teval):
        self.pars = pars
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
        # Concentration of composites (gCOD/L)
        Ss = x[1]
        # Concentration of particulate inerts (gCOD/L)
        alk = x[2]
        # Concentration of soluble inerts (gCOD/L)
        o2 = x[3]
        # Concentration of carbohydrates (gCOD/L)
        nh4 = x[4]
        # Concentration of proteins (gCOD/L)
        no2 = x[5]
        # Concentration of lipids (gCOD/L)
        no3 = x[6]
        # Concentration of total butirate (gCOD/L)
        x_S_film = x[7]
        # Concentration of total propionate (gCOD/L)
        x_I_film = x[8]
        # Concentration of glucose degraders in solid phase (gCOD/L)
        x_H_film = x[9]
        # Concentration of lcfa degraders in solid phase (gCOD/L)
        x_A_film = x[10]
        # Concentration of valerate degraders in solid phase (gCOD/L)
        x_N_film = x[11]
        # Concentration of aminoacids (gCOD/L) 
        x_sto_film = x[12]
        # Solid volume [L]
        V_sol = x[13]
        # Average bioparticle diameter [dm]
        dbp = x[14]
        # Energy consumed by agitation
        Ec_ag = x[15]
        # Energy consumed by aireation
        Ec_air = x[16]
        # Cumulative energy consumed
        Ec = x[17]
        # Soluble inerts (gCOD/L)
        Si = x[18]

        

        # Inputs (3):
        #  pars  = kinetics parameters (31)
         #  Aerobic endogenous respiration rate of x_A
        b_A = self.pars[0]
        #   Aerobic endogenous respiration rate of x_N
        b_N = self.pars[1]
        #   Aerobic endogenous respiration rate of x_H
        b_H = self.pars[2]
        #   Aerobic endogenous respiration rate of x_sto
        b_sto = self.pars[3]        
        #   Saturation constant for x_S
        ks_X = self.pars[4]
        #   Reduction factor for anoxic activity
        n_g = self.pars[5]
        #   Saturation constant for alk
        k_alk = self.pars[6]
        #   Oxygen mass transfer coefficient
        kLa = self.pars[7]        
        #   Saturation constant for x_sto
        ks_sto = self.pars[8]
        #   Saturation constant for Ss
        ks_ss = self.pars[9]        
        #   nh4 saturation constant for x_A
        k_nh4 = self.pars[10]
        #   no2 saturation constant for x_A
        kA_no2 = self.pars[11]        
        #   no3 saturation constant for x_A 
        kA_no3 = self.pars[12]
        #   no2 saturation constant for x_N
        kN_no2 = self.pars[13]
        #   no3 saturation constant for x_N
        kN_no3 = self.pars[14]
        #   no2 saturation constant for x_H
        kh_no2 = self.pars[15]    
        #   no3 saturation constant for x_H
        kh_no3 = self.pars[16]
        #   o2 saturation constant for x_A
        k_A_o2 = self.pars[17]
        #   o2 saturation constant for x_N
        k_N_o2 = self.pars[18]
        #   o2 saturation constant for x_H
        k_H_o2 = self.pars[19]
        #   Maximum growth rate of x_A
        umax_A = self.pars[20]
        #   Maximum growth rate of x_N
        umax_N = self.pars[21]
        #   Maximum growth rate of x_H
        umax_H = self.pars[22]
        #   Storage rate constant
        k_sto = self.pars[23]
        #   Yield of x_A per nh4
        y_A = self.pars[24]
        #   Yield of x_N per no2
        y_N = self.pars[25]
        #   Yield of x_H per Ss
        y_H = self.pars[26]
        #   Yield of x_H per s_sto
        y_hsto = self.pars[27]
        #   Yield coefficient for storage on substrate
        y_sto = self.pars[28]
        #   Hydrolysis of particulate substrate constant
        k_h_s = self.pars[29]
        #   Inhibition constant
        ki_no2 = self.pars[30]
        
        

        #  cts   = Constants (15)
        #    Inlet flow of raw substrate
        qby = self.cts[0]
        #    Inlet flow of anaerobic digester
        qan = self.cts[1]
        #   Outlet flow
        qout = self.cts[2]
        #   Gas volume 
        V_gas = self.cts[3]
        #   Outlet solid flow
        qs = self.cts[4]
        #   Mixing constant
        lamb = self.cts[5]
        #   Bioparticle-related constants
        rho_s = self.cts[6]
        AC_bp = self.cts[7]
        MC_bp = self.cts[8]
        k_det = self.cts[9]
        #   Saturation oxygen concentration
        o2_s = self.cts[10]
        #   Production of x_I in endogenous respiration
        f_X = self.cts[11]
        #   N content of x_I
        f_N_I = self.cts[12]
        #   N content of biomass x_H, x_N, x_A
        f_N_B = self.cts[13]
        #   Production of s_I on Ss
        f_S_I = self.cts[14]
        #   N content of Ss
        f_N_Ss = self.cts[15]

        #  feed = Feed Concentration (23)
        # inlet of alk from anaerobic digester (mol/L)
        alk_an = self.feed[0]
        # inlet of nh4 from anaerobic digester (gN/L)
        nh4_an = self.feed[1]
        # inlet of particulate substre from anaerobic digester (gCOD/L)
        x_S_an = self.feed[2]
        # inlet of soluble substrate from anaerobic digester (gCOD/L)
        Ss_an = self.feed[3]
        # inlet of alk from raw substrate (mol/L)
        alk_by = self.feed[4]
        # inlet of nh4 from raw substrate (gN/L)
        nh4_by = self.feed[5]
        # inlet of particulate substrate from raw substrate (gCOD/L)
        x_S_by = self.feed[6]
        # inlet of soluble substrate from raw substrate (gCOD/L)
        Ss_by = self.feed[7]
        # inlet of oxygen from raw substrate (mol/L)
        o2_in = self.feed[8]
        # inlet of nitrite from raw substrate (gN/L)
        no2_in = self.feed[9]
        # inlet of nitrate from raw substrate (gN/L)
        no3_in = self.feed[10]

        V_total = V_liq + V_gas + V_sol
        el = V_liq/V_total
        es = V_sol/V_total
        eg = V_gas/V_total

        # Growth rate equations
        hydrolysis = k_h_s * ((x_S_film/x_H_film)/(ks_X+(x_S_film/x_H_film)))
        u_2 = umax_H * (o2/(k_H_o2+o2)) * (Ss/(ks_ss+Ss))
        u_3 = umax_H * (o2/(k_H_o2+o2)) * (Ss/(ks_ss+Ss)) * ((x_sto_film/x_H_film)/(ks_sto+(x_sto_film/x_H_film)))
        u_4_a = umax_H * n_g * (k_H_o2/(k_H_o2+o2)) * (Ss/(ks_ss+Ss))
        u_4 = umax_H * n_g * (k_H_o2/(k_H_o2+o2)) * (no2/(kh_no2+no2)) * (Ss/(ks_ss+Ss))
        u_5 = umax_H * n_g * (k_H_o2/(k_H_o2+o2)) * (no3/(kh_no3+no3)) * (Ss/(ks_ss+Ss))
        u_6 = umax_H * n_g * (k_H_o2/(k_H_o2+o2)) * (no2/(kh_no2+no2)) * ((x_sto_film/x_H_film)/(ks_sto+(x_sto_film/x_H_film)))       
        u_7 = umax_H * n_g * (k_H_o2/(k_H_o2+o2)) * (no3/(kh_no3+no3)) * ((x_sto_film/x_H_film)/(ks_sto+(x_sto_film/x_H_film)))  
        u_8 = b_H * (o2/(k_H_o2+o2)) 
        u_9 = b_H  * n_g * (k_H_o2/(k_H_o2+o2)) * (no2/(kh_no2+no2))
        u_10 = b_H  * n_g * (k_H_o2/(k_H_o2+o2)) * (no3/(kh_no3+no3))
        u_11 = k_sto * (o2/(k_H_o2+o2)) * (Ss/(ks_ss+Ss))
        u_12 = k_sto * n_g * (k_H_o2/(k_H_o2+o2)) * (no2/(kh_no2+no2)) * (Ss/(ks_ss+Ss))
        u_13 = k_sto * n_g * (k_H_o2/(k_H_o2+o2)) * (no3/(kh_no3+no3)) * (Ss/(ks_ss+Ss))
        u_14 = b_sto * (o2/(k_H_o2+o2))
        u_15 = b_sto * n_g * (k_H_o2/(k_H_o2+o2)) * (no2/(kh_no2+no2))
        u_16 = b_sto * n_g * (k_H_o2/(k_H_o2+o2)) * (no3/(kh_no3+no3))
        u_17 = umax_A * (o2/(k_A_o2+o2)) * (nh4/(k_nh4+nh4)) * (alk/(k_alk+alk)) * (1/(1+(k_nh4/nh4)+(no2/ki_no2)))
        u_18 = b_A * (o2/(k_A_o2+o2))
        u_19 = b_A * n_g * (k_A_o2/(k_A_o2+o2)) * (no2/(kA_no2+no2))
        u_20 = b_A * n_g * (k_A_o2/(k_A_o2+o2)) * (no3/(kA_no3+no3))
        u_21 = umax_N * (o2/(k_N_o2+o2)) * (no2/(kN_no2+no2)) * (1/(1+(kN_no2/no2)+(no2/ki_no2)))
        u_22 = b_N * (o2/(k_N_o2+o2))
        u_23 = b_N * n_g * (k_N_o2/(k_N_o2+o2)) * (no2/(kN_no2+no2))
        u_24 = b_N * n_g * (k_N_o2/(k_N_o2+o2)) * (no3/(kN_no3+no3))


        # Other algebraic equations
        rhet_s = lamb*(es*((u_2+u_3+u_4+u_4_a+u_5+u_6+u_7-u_8-u_9-u_10)*x_H_film + (-(1/y_hsto)*u_3-(1/y_hsto)*u_6-(1/y_hsto)*u_7+u_11+u_12+u_13-u_14-u_15-u_16)*x_sto_film + (u_17-u_18+u_19+u_20)*x_A_film + (u_21-u_22-u_23-u_24)*x_N_film + (u_8+u_9+u_10+u_18+u_19+u_20+u_22+u_23+u_24)*f_X*x_I_film - k_det*(x_H_film+x_A_film+x_N_film+x_sto_film+x_I_film)))
        qin = qan + qby

        if qin == 0:
            Ss_in = 0
            x_S_in = 0
            nh4_in = 0
            alk_in = 0
        else:
            Ss_in = (qan*Ss_an + qby*Ss_by)/qin
            x_S_in = (qan*x_S_an + qby*x_S_by)/qin
            nh4_in = (qan*nh4_an + qby*nh4_by)/qin
            alk_in = (qan*alk_an + qby*alk_by)/qin

        # Mass balance: volume derivative
        dV_liq = qin - qout
        dV_sol = -qs + ((V_total*rhet_s*(113/160)) / (rho_s*(1-AC_bp/100)*(1-MC_bp/100)))
        deriv_el = (dV_liq*V_total-(dV_liq+dV_sol)*V_liq)/(V_total**2)
        deriv_eg = -(V_gas*(dV_liq+dV_sol))/(V_total**2)
        # Species balance: concentration derivative
        # Chain rule: d(V*Ca)/dt = Ca * dV/dt + V * dCa/dt
        # Soluble substrate mass balance
        dSsdt = (qin*Ss_in- qout*Ss)/(V_total*el) + lamb*((1-f_S_I)*hydrolysis*(x_H_film*(es/el)) - (1/y_H)*u_2*(x_H_film*(es/el)) - (1/y_H)*u_4_a*(x_H_film*(es/el)) - (1/y_H)*u_4*(x_H_film*(es/el)) - (1/y_H)*u_5*(x_H_film*(es/el)) - (1/y_sto)*u_11*(x_H_film*(es/el)) - (1/y_sto)*u_12*(x_H_film*(es/el)) - (1/y_sto)*u_13*(x_H_film*(es/el))) - (Ss*(dV_liq+dV_sol)/V_total) - (deriv_el*Ss)/el 
        # Inert soluble substrate mass balance
        dSidt = lamb*(f_S_I*hydrolysis*(x_H_film*(es/el)))
        # Particulate inerts mass balance
        dalkdt = (qin*alk_in - qout*alk)/(V_total*el) + lamb*((1/(64*y_H)-f_N_B/14)*u_2*(x_H_film*(es/el)) - f_N_B/14*u_3*(x_H_film*(es/el)) + (1/(64*y_H)+(1/y_H-1)/24-f_N_B/14)*u_4*(x_H_film*(es/el)) + (1/(64*y_H)+(1/y_H-1)/40-f_N_B/14)*u_5*(x_H_film*(es/el)) + ((1/y_sto-1)/24-f_N_B/14)*u_6*(x_H_film*(es/el)) + ((1/y_sto-1)/24-f_N_B/14)*u_7*(x_H_film*(es/el)) + (f_N_B-f_N_I*f_X)/14*u_8*(x_H_film*(es/el)) + ((f_N_B-f_N_I*f_X)/14+(1-f_X)/24)*u_9*(x_H_film*(es/el)) + ((f_N_B-f_N_I*f_X)/14+(1-f_X)/40)*u_10*(x_H_film*(es/el)) + (1/(64*y_sto))*u_11*(x_H_film*(es/el)) + (1/(64*y_sto)+(1/y_sto-1)/24)*u_12*(x_H_film*(es/el)) + (1/(64*y_sto)+(1/y_sto-1)/40)*u_13*(x_H_film*(es/el)) + (1/24)*u_15*(x_sto_film*(es/el)) + (1/40)*u_16*(x_sto_film*(es/el)) - ((f_N_B + 2/y_A)/14)*u_17*(x_A_film*(es/el)) + ((f_N_B-f_N_I*f_X)/14)*u_18*(x_A_film*(es/el)) + ((f_N_B-f_N_I*f_X)/14+(1-f_N_B)/24)*u_19*(x_A_film*(es/el)) + ((f_N_B-f_N_I*f_X)/14+(1-f_N_B)/40)*u_20*(x_A_film*(es/el)) - (f_N_B/14)*u_21*(x_N_film*(es/el)) - ((f_N_B-f_N_I*f_X)/14)*u_22*(x_N_film*(es/el)) + ((f_N_B-f_N_I*f_X)/14+(1-f_N_B)/24)*u_23*(x_N_film*(es/el)) + ((f_N_B-f_N_I*f_X)/14+(1-f_N_B)/40)*u_24*(x_N_film*(es/el))) - (alk*(dV_liq+dV_sol)/V_total) - (deriv_el*alk)/el 
        # Inerts carbohydrates mass balance
        do2dt = (qin*o2_in - qout*o2)/(V_total*el) + lamb*((-1/y_H-1)*u_2*(x_H_film*(es/el)) + (-1/y_hsto-1)*u_3*(x_H_film*(es/el)) - (-1-f_X)*u_8*(x_H_film*(es/el)) - (1/y_sto -1)*u_11*(x_H_film*(es/el)) - u_14*(x_sto_film*(es/el)) - (3.43/y_A -1)*u_17*(x_A_film*(es/el)) - (1-f_X)*u_18*(x_A_film*(es/el)) - (1.14/y_N)*u_21*(x_N_film*(es/el)) - (1-f_X)*u_22*(x_N_film*(es/el)) + kLa*(o2_s - o2)) - (o2*(dV_liq+dV_sol)/V_total) - (deriv_el*o2)/el 
        # carbohydrates mass balance
        dnh4dt = (qin*nh4_in - qout*nh4)/(V_total*el) + lamb*(-f_N_B*u_2*(x_H_film*(es/el)) + (f_N_Ss-f_N_B)*u_4_a*(x_H_film*(es/el)) - f_N_B*u_3*(x_H_film*(es/el)) - f_N_B*u_4*(x_H_film*(es/el)) - f_N_B*u_5*(x_H_film*(es/el)) - f_N_B*u_6*(x_H_film*(es/el)) - f_N_B*u_7*(x_H_film*(es/el)) + (f_N_B-f_N_I*f_X)*u_8*(x_H_film*(es/el)) + (f_N_B-f_N_I*f_X)*u_9*(x_H_film*(es/el)) + (f_N_B-f_N_I*f_X)*u_10*(x_H_film*(es/el)) - f_N_Ss*u_11*(x_H_film*(es/el)) - f_N_Ss*u_12*(x_H_film*(es/el)) - f_N_Ss*u_13*(x_H_film*(es/el)) - (f_N_B+1/y_A)*u_17*(x_A_film*(es/el)) + (f_N_B-f_N_I*f_X)*u_18*(x_A_film*(es/el)) + (f_N_B-f_N_I*f_X)*u_19*(x_A_film*(es/el)) + (f_N_B-f_N_I*f_X)*u_20*(x_A_film*(es/el)) - f_N_B*u_21*(x_N_film*(es/el)) + (f_N_B-f_N_I*f_X)*u_22*(x_N_film*(es/el)) + (f_N_B-f_N_I*f_X)*u_23*(x_N_film*(es/el)) + (f_N_B-f_N_I*f_X)*u_24*(x_N_film*(es/el))) - (nh4*(dV_liq+dV_sol)/V_total) - (deriv_el*nh4)/el
        # protein mass balance
        dno2dt = (qin*no2_in - qout*no2)/(V_total*el) + lamb*(-((1/y_H -1)/1.71)*u_4*(x_H_film*(es/el)) - ((1/y_hsto -1)/1.71)*u_6*(x_H_film*(es/el)) - ((1-f_X)/1.71)*u_9*(x_H_film*(es/el)) - ((1/y_sto -1)/1.71)*u_12*(x_H_film*(es/el)) - (1/1.71)*u_15*(x_sto_film*(es/el)) + (1/y_A)*u_17*(x_A_film*(es/el)) - ((1-f_X)/1.71)*u_19*(x_A_film*(es/el)) - (1/y_N)*u_21*(x_N_film*(es/el)) - ((1-f_X)/2.68)*u_24*(x_N_film*(es/el))) - (no2*(dV_liq+dV_sol)/V_total) - (deriv_el*no2)/el
        # lipids mass balance
        dno3dt = (qin*no3_in - qout*no3)/(V_total*el) + lamb*(-((1/y_H-1)/2.86)*u_5*(x_H_film*(es/el)) - ((1/y_hsto-1)/2.86)*u_7*(x_H_film*(es/el)) - ((1-f_X)/2.86)*u_10*(x_H_film*(es/el)) - ((1/y_sto-1)/2.86)*u_13*(x_H_film*(es/el)) - (1/2.86)*u_16*(x_sto_film*(es/el)) - ((1-f_X)/2.86)*u_20*(x_A_film*(es/el)) + (1/y_N)*u_21*(x_N_film*(es/el)) - ((1-f_X)/2.86)*u_24*(x_N_film*(es/el))) - (no3*(dV_liq+dV_sol)/V_total) - (deriv_el*no3)/el
        # glucose mass balance
        dx_H_filmdt = (- qs*x_H_film)/(V_total*es) + lamb*(u_2*x_H_film+u_3*x_H_film+u_4*x_H_film+u_5*x_H_film+u_6*x_H_film+u_7*x_H_film-u_8*x_H_film-u_9*x_H_film-u_10*x_H_film - k_det*x_H_film) - (x_H_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_H_film)/es
        # aminoacids degraders mass balance
        dx_sto_filmdt = (- qs*x_sto_film)/(V_total*es) + lamb*(-(1/y_hsto)*u_3*x_H_film - (1/y_hsto)*u_6*x_H_film - (1/y_hsto)*u_7*x_H_film + u_11*x_H_film + u_12*x_H_film + u_13*x_H_film - u_14*x_sto_film - u_15*x_sto_film - u_16*x_sto_film - k_det*x_sto_film) - (x_sto_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_sto_film)/es
        # lcfa degraders mass balance
        dx_A_filmdt = (- qs*x_A_film)/(V_total*es) + lamb*(u_17*x_A_film - u_18*x_A_film - u_19*x_A_film - u_20*x_A_film - k_det*x_A_film) - (x_A_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_A_film)/es
        # valerate degraders mass balance
        dx_N_filmdt = (- qs*x_N_film)/(V_total*es) + lamb*(u_21*x_N_film - u_22*x_N_film - u_23*x_N_film - u_24*x_N_film - k_det*x_N_film) - (x_N_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_N_film)/es
        # propionic degraders mass balance
        dx_I_filmdt = (- qs*x_I_film)/(V_total*es) + lamb*(f_X*u_8*x_H_film + f_X*u_9*x_H_film + f_X*u_10*x_H_film + f_X*u_18*x_A_film + f_X*u_19*x_A_film + f_X*u_20*x_A_film + f_X*u_22*x_N_film + f_X*u_23*x_N_film + f_X*u_24*x_N_film) - (x_I_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_I_film)/es
        # acetate degraders mass balance
        dx_S_filmdt = (qin*x_S_in - qs*x_S_film)/(V_total*es) + lamb*(-hydrolysis*x_H_film) - (x_S_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_S_film)/es
        # average bioparticle diameter
        if lamb == 0:
            dbpdt = 0
        else:
            dbpdt = (dbp/3)*(rhet_s/(es*(x_H_film+x_sto_film+x_I_film+x_N_film+x_A_film)))

        # Energy consumed by agitation
        dEc_agdt = lamb*0.08918826448*0.65*24

        # Energy consumed by aireation
        dEc_airdt = lamb*((0.03*1000*9.8*0.46)/3600)*24

        # Cumulative energy consumed
        dEcdt = dEc_agdt + dEc_airdt
        
        f = np.array([dV_liq, dSsdt, dalkdt, do2dt, dnh4dt, dno2dt, dno3dt, dx_S_filmdt, dx_I_filmdt, dx_H_filmdt, dx_A_filmdt, dx_N_filmdt, dx_sto_filmdt, dV_sol, dbpdt, dEc_agdt, dEc_airdt, dEcdt, dSidt])
        
        # Return derivatives
        return f

    # Function that solves the DAE system
    def ode(self):
        
        # Differential equations
        states = solve_ivp(self.vessel, self.tspan, self.initial, t_eval=self.teval, method='BDF', rtol=1E-9, atol=1E-12)
        
        time = states.t
        v_liq = states.y[0]
        sol_substrate = states.y[1]
        alkalinity = states.y[2]
        oxygen = states.y[3]
        ammonia = states.y[4]
        nitrite = states.y[5]
        nitrate = states.y[6]
        part_substrate = states.y[7]
        part_inerts = states.y[8]
        biomass_H_film = states.y[9]
        biomass_A_film = states.y[10]
        biomass_N_film = states.y[11]
        sto = states.y[12]
        v_sol = states.y[13]
        dbp = states.y[14]
        Ec_agit = states.y[15]
        Ec_air = states.y[16]
        Ec = states.y[17]
        sol_inerts = states.y[18]

        # Algebraic equations
        
        COD_Total = (v_sol/v_liq)*(biomass_H_film + biomass_A_film + biomass_N_film + sto + part_inerts + part_substrate) + sol_substrate

        COD_Soluble = sol_substrate + sol_inerts

        NAT = ammonia*1000

        no3 = nitrate*1000

        V_total = v_liq + self.cts[3] + v_sol

        return time, v_liq, sol_substrate, alkalinity, oxygen, ammonia, nitrite, nitrate, part_substrate, part_inerts, biomass_H_film, biomass_A_film, biomass_N_film, sto, v_sol, dbp, COD_Total, COD_Soluble, NAT, no3, V_total, Ec_agit, Ec_air, Ec, sol_inerts
