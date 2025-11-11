import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, root
import Algebraic_equations as ae

# The Tanque_Digestor class implement the Angelidaki et al. (1999) model of Anaerobic Digestion

class Tanque_Digestor():
    def __init__(self, pars, cts, feed, initial, tspan, teval):
        self.pars = pars
        self.cts = cts
        self.feed = feed
        self.initial = initial
        self.tspan = tspan
        self.teval = teval
    
    algebraic_vars = []
    tie = []
    
    #CSTR's FUNCTION - Differential-Algebraic Equation System (DAE)
    def vessel(self, t, x):
        # States (26):
        # Liquid volume (L)
        V_liq = x[0]
        # Concentration of composites (gCOD/L)
        x_c = x[1]
        # Concentration of particulate inerts (gCOD/L)
        x_i = x[2]
        # Concentration of soluble inerts (gCOD/L)
        s_i = x[3]
        # Concentration of carbohydrates (gCOD/L)
        ch = x[4]
        # Concentration of proteins (gCOD/L)
        prot = x[5]
        # Concentration of lipids (gCOD/L)
        lip = x[6]
        # Concentration of glucose (gCOD/L)
        gl = x[7]
        # Concentration of aminoacids (gCOD/L) 
        aa = x[8]
        # Concentration of Long Chain Fatty Acids (gCOD/L)
        lcfa = x[9]
        # Concentration of total valerate (gCOD/L)
        va_tot = x[10]
        # Concentration of total butirate (gCOD/L)
        bu_tot = x[11]
        # Concentration of total propionate (gCOD/L)
        pro_tot = x[12]
        # Concentration of total acetate (gCOD/L)
        ac_tot = x[13]
        # Concentration of soluble hydrogen (mol/L)
        h2 = x[14]
        # Concentration of hydrogen gas (mol/L)
        h2_g = x[15]
        # Concentration of liquid methane (gCOD/L)
        ch4 = x[16]
        # Concentration of gas methane (gCOD/L)
        ch4_g = x[17]
        # Concentration of inorganic carbon (mol/L)
        i_c = x[18]
        # Concentration of gas carbon dioxide (mol/L)
        co2_g = x[19]
        # Concentration of inorganic nitrogen (mol/L)
        i_n = x[20]
        # Concentration of glucose degraders (gCOD/L)
        x_gl = x[21]
        # Concentration of aminoacids degraders (gCOD/L)
        x_aa = x[22]
        # Concentration of lcfa degraders (gCOD/L) 
        x_lcfa = x[23]
        # Concentration of valerate and butyrate degraders (gCOD/L)
        x_c4 = x[24]
        # Concentration of propionate degraders (gCOD/L)
        x_pro = x[25]
        # Concentration of acetate degraders (gCOD/L)
        x_ac = x[26]
        # Concentration of hydrogen degraders (gCOD/L)
        x_h2 = x[27]
        # Solid volume [L]
        V_sol = x[28]
        # Concentration of glucose degraders in solid phase (gCOD/L)
        x_gl_film = x[29]
        # Concentration of aminoacids degraders in solid phase (gCOD/L)
        x_aa_film = x[30]
        # Concentration of lcfa degraders in solid phase (gCOD/L)
        x_lcfa_film = x[31]
        # Concentration of valerate degraders in solid phase (gCOD/L)
        x_c4_film = x[32]
        # Concentration of propionate degraders in solid phase (gCOD/L)
        x_pro_film = x[33]
        # Concentration of acetate degraders in solid phase (gCOD/L)
        x_ac_film = x[34]
        # Concentration of acetate degraders in solid phase (gCOD/L)
        x_h2_film = x[35]
        # Average bioparticle diameter [dm]
        dbp = x[36]
        # Concentration of cations [mol/L]
        cat = x[37]
        # Concentration of cations [mol/L]
        an = x[38]
        # Cumulative methane [L]
        cum_methane = x[39]
        # Cumulative biogas [L]
        cum_biogas = x[40]

        # Inputs (3):
        #  pars  = kinetics parameters (47)
         # dissagregation of composites
        k_dis = self.pars[0]
        #   hydrolysis of carbohydrates
        k_hyd_ch = self.pars[1]
        #   hydrolysis of proteins
        k_hyd_pro = self.pars[2]
        #   hydrolysis of carbohydrates
        k_hyd_lip = self.pars[3]        
        #   death of biomass
        k_dead = self.pars[4]
        #   Ks of inorganic nitrogen
        ks_in = self.pars[5]
        #   Ki of LCFA for hydrogen uptake
        ki_h2_lcfa = self.pars[6]
        #   Ki of acetate on valerate consumption
        ki_h2_c4 = self.pars[7]
        #   Ki of acetate on butirate consumption
        ki_h2_pro = self.pars[8]
        #   Ki of acetate on propionate consumption
        ki_fa = self.pars[9]
        #   Ki of NH3
        ki_nh3 = self.pars[10]
        #   pH upper limit acidogenesis/acetogenesis
        pH_ul_ac = self.pars[11]        
        #   pH lower limit acidogenesis/acetogenesis
        pH_ll_ac = self.pars[12]
        #   pH upper limit hydrogen methanogenic
        pH_ul_meth2 = self.pars[13]        
        #   pH lower limit hydrogen methanogenic 
        pH_ll_meth2 = self.pars[14]
        #   pH upper limit acetoclastic methanogenic
        pH_ul_metac = self.pars[15]        
        #   pH lower limit acetoclastic methanogenic 
        pH_ll_metac = self.pars[16]
        #   Acidogenesis of glucose
        umax_acid_su = self.pars[17]
        ks_gl = self.pars[18]
        y_gl = self.pars[19]    
        #   Acidogenesis of aminoacids
        umax_acid_aa = self.pars[20]
        ks_aa = self.pars[21]
        y_aa = self.pars[22]
        #   Acetogenesis of lcfa
        umax_acid_lcfa = self.pars[23]
        ks_lcfa = self.pars[24]
        y_lcfa = self.pars[25]
        #   Acetogenesis of valerate
        umax_acet_c4 = self.pars[26]
        ks_c4 = self.pars[27]
        y_c4 = self.pars[28]
        #   Acetogenesis of propionate
        umax_acet_pro = self.pars[29]
        ks_pro = self.pars[30]
        y_pro = self.pars[31]
        #   Methanogenesis of hydrogen
        umax_met_h2 = self.pars[32]
        ks_h2 = self.pars[33]
        y_h2 = self.pars[34]
        #   Methanogenesis of acetate
        umax_met_ac = self.pars[35]
        ks_ac = self.pars[36]
        y_ac = self.pars[37]
        # Acid base reactions
        deltaH_CH4 = self.pars[38]
        deltaH_CO2 = self.pars[39]
        deltaH_H2 = self.pars[40]
        temp = self.pars[41]
        #   Henry's constants
        kH298_ch4 = self.pars[42]
        kH298_co2 = self.pars[43]
        kH298_h2 = self.pars[44]
        #   Mass transfer coefficient   
        kLa = self.pars[45]

        #  cts   = Constants (36)
        #   Conversion
        #   inerts of carbohydrates
        f_xi_xc = self.cts[0] 
        #   inerts of proteins
        f_si_xc = self.cts[1]
        #   Carbohydrates of composites
        f_xch_xc = self.cts[2]
        #   Proteins of composites
        f_xpro_xc = self.cts[3]
        #   Lipids of composites
        f_xlip_xc = self.cts[4]
        #   LCFA of lipids
        f_lip_lcfa = self.cts[5]
        #   Butirate of glucose
        f_bu_gl = self.cts[6]
        #   Propionate of glucose
        f_pro_gl = self.cts[7]
        #   Acetate of glucose
        f_ac_gl = self.cts[8]
        #   Valerate of aminoacids
        f_va_aa = self.cts[9]
        #   Butirate of aminoacids
        f_bu_aa = self.cts[10]
        #   Propionate of aminoacids
        f_pro_aa = self.cts[11]
        #   Acetate of aminoacids
        f_ac_aa = self.cts[12] 
        #   Acetate of LCFA      
        f_ac_lcfa = self.cts[13]
        #   Propionate of valerate
        f_pro_va = self.cts[14]
        #   Acetate of valerate
        f_ac_va = self.cts[15]
        #   Acetate of butirate
        f_ac_bu = self.cts[16]
        #   Acetate of propionate
        f_ac_pro = self.cts[17]
        #   Hydrogen of sugars
        f_h2_gl = self.cts[18]
        #   Hydrogen of aminoacids
        f_h2_aa = self.cts[19]
        #   Hydrogen of LCFA
        f_h2_lcfa = self.cts[20]
        #   Hydrogen of Valerate
        f_h2_va = self.cts[21]
        #   Hydrogen of butyrate
        f_h2_bu = self.cts[22]
        #   Hydrogen of propionate
        f_h2_pro = self.cts[23]

        #   Flows
        #    inlet flow
        qin = self.cts[24]
        #    outlet flow
        qout = self.cts[25]

        #   Nitrogen and carbon conversion
        #   Biomass to nitrogen factor
        n_bio = self.cts[26]
        #   Aminoacids to nitrogen factor
        n_aa = self.cts[27]
        #   hydrogen to carbon factor
        c_h2 = self.cts[28]
        #   glucose to carbon factor
        c_gl = self.cts[29]
        #   aminoacids to carbon factor
        c_aa = self.cts[30]
        #   biomass to carbon factor
        c_bm = self.cts[31]
        #   valerate to carbon factor
        c_va = self.cts[32]
        #   butyrate to carbon factor
        c_bu = self.cts[33]
        #   propionate to carbon factor
        c_pro = self.cts[34]
        #   acetate to carbon factor 
        c_ac = self.cts[35]
        #   methane to carbon factor 
        c_ch4 = self.cts[36]

        #   Others 
        V_gas = self.cts[37]
        #    Outlet solid flow
        qs = self.cts[38]
        # Mixing constant
        lamb = self.cts[39]
        # Bioparticle-related constants
        rho_s = self.cts[40]
        AC_bp = self.cts[41]
        MC_bp = self.cts[42]
        k_det = self.cts[43]

        #  feed = Feed Concentration (23)
        # inlet of inerts carbohydrates (gCOD/L)
        x_i_in = self.feed[0]
        # inlet of inerts proteins (gCOD/L)
        s_i_in = self.feed[1]
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
        # Inlet of cations [mol/L]
        cat_in = self.feed[14]
        # Inlet of anions [mol/L]
        an_in = self.feed[15]
        
        
        # Acid base reactions for pH calculation
        ka_nh3 = 10**(-9.25)*math.exp(0.07*(temp-298.15))
        ka_co2_1 = 10**(-6.35)*math.exp(0.01*(temp-298.15))
        ka_co2_2 = 10**(-10.30)*math.exp(0.005*(temp-298.15))
        ka_lcfa = 10**(-4.25)
        ka_va = 10**(-4.86)
        ka_pro = 10**(-4.94)
        ka_bu = 10**(-4.92)
        ka_ac = 10**(-4.81)
        ka_w = 10**(-14)*math.exp(0.076*(temp-298.15))

        def charge_balance(y):
            return cat + ((i_n*y)/(ka_nh3+y)) + y - (i_c/(1+(y/ka_co2_2)*(y/ka_co2_1)+(y/ka_co2_2))) - 2*(i_c/((ka_co2_2/y)+(y/ka_co2_1)+1)) - ((lcfa - ((lcfa*y)/(ka_lcfa+y)))/736) - ((ac_tot - ((ac_tot*y)/(ka_ac+y)))/64) - ((pro_tot - ((pro_tot*y)/(ka_pro+y)))/112) - ((bu_tot - ((bu_tot*y)/(ka_bu+y)))/160) - ((va_tot - ((va_tot*y)/(ka_va+y)))/208) - (ka_w/y) - an        


        s_h = float(fsolve(charge_balance, 1E-7))
        #print(s_h)
        co2_liq = (i_c/((ka_co2_2/s_h)+(s_h/ka_co2_1)+1))*(s_h/ka_co2_1)
                                                                             
        
        pH = ae.pH1(s_h)
        #print(pH)
        
        nh3 = (i_n - ((i_n*s_h)/(ka_nh3+s_h)))*17.031
        S_hac = (s_h*ac_tot)/(ka_ac+s_h)

        # Pressures
        p_ch4 = ae.pres(ch4_g, temp)
        p_co2 = ae.pres(co2_g, temp)
        p_h2 = ae.pres(h2_g, temp)
        p_h2o = 0.0313*math.exp(5290*((1/298.15)-(1/temp)))
        p_tot = 1.013


        # Transfer terms
        kH_ch4 = ae.kH_temp(kH298_ch4, deltaH_CH4, temp)
        kH_co2 = ae.kH_temp(kH298_co2, deltaH_CO2, temp)
        kH_h2 = ae.kH_temp(kH298_h2, deltaH_H2, temp)
        rho_ch4 = ae.rho_gas(lamb, kLa, ch4, kH_ch4*64, p_ch4)
        rho_co2 = ae.rho_gas(lamb, kLa, co2_liq, kH_co2, p_co2)
        rho_h2 = ae.rho_gas(lamb, kLa, h2, kH_h2*16, p_h2)


        # Temperature dependence
        umax_acid_su_Te = umax_acid_su #- 0.0936*(55 - (temp-273.15))
        umax_acid_aa_Te = umax_acid_aa #- 0.1176*(55 - (temp-273.15))
        umax_met_h2_Te = umax_met_h2 #- 0.00984*(55 - (temp-273.15))
        umax_acid_lcfa_Te = umax_acid_lcfa #- 0.01008*(55 - (temp-273.15))
        umax_acet_pro_Te = umax_acet_pro #- 0.00888*(53 - (temp-273.15))
        umax_acet_c4_Te = umax_acet_c4 #- 0.01248*(60 - (temp-273.15))
        umax_met_ac_Te = umax_met_ac #- 0.01128*(55 - (temp-273.15))

        V_total = V_liq + V_gas + V_sol
        el = V_liq/V_total
        es = V_sol/V_total
        eg = V_gas/V_total

        ki_lcfa = (ki_fa*(x_lcfa+x_lcfa_film*(es/el)))/lcfa

        # Growth rate equations
        u_acid_su = umax_acid_su_Te * (1/(1+(ks_gl/gl))) * ae.inhibition2(ks_in, i_n) * ae.inhibitionpH2(pH_ll_ac, pH_ul_ac, pH)
        u_acid_aa = umax_acid_aa_Te * (1/(1+(ks_aa/aa))) * ae.inhibitionpH2(pH_ll_ac, pH_ul_ac, pH)
        u_met_h2 = umax_met_h2_Te * (1/(1+(ks_h2/h2))) * ae.inhibition2(ks_in, i_n) * ae.inhibition1(ki_h2_lcfa, lcfa) * ae.inhibitionpH2(pH_ll_meth2, pH_ul_meth2, pH)
        u_acid_lcfa = umax_acid_lcfa_Te * (1/(ks_lcfa + lcfa + ((lcfa**2)/ki_lcfa))) * ae.inhibition2(ks_in, i_n) * ae.inhibitionpH(pH_ll_ac, pH_ul_ac, pH)
        u_acet_val = umax_acet_c4_Te * (va_tot/(ks_c4+va_tot)*(1/(1+(bu_tot/va_tot)))) * ae.inhibition2(ks_in, i_n) * ae.inhibition1(ki_h2_c4, h2) * ae.inhibitionpH(pH_ll_ac, pH_ul_ac, pH)
        u_acet_but = umax_acet_c4_Te * (bu_tot/(ks_c4+bu_tot)*(1/(1+(va_tot/bu_tot)))) * ae.inhibition2(ks_in, i_n) * ae.inhibition1(ki_h2_c4, h2) * ae.inhibitionpH(pH_ll_ac, pH_ul_ac, pH)        
        u_acet_pro = umax_acet_pro_Te * (1/(1+(ks_pro/pro_tot))) * ae.inhibition2(ks_in, i_n) * ae.inhibition1(ki_h2_pro, h2) * ae.inhibitionpH(pH_ll_ac, pH_ul_ac, pH)  
        u_met_ac = umax_met_ac_Te * (1/(1+(ks_ac/ac_tot))) * ae.inhibition2(ks_in, i_n) * ae.inhibition1(ki_lcfa, lcfa) * ae.inhibition1(ki_nh3, nh3) * ae.inhibitionpH(pH_ll_metac, pH_ul_metac, pH)


        # Other algebraic equations
        rhet_s = lamb*(es*(u_acid_su*x_gl_film + u_acid_aa*x_aa_film + u_met_h2*x_h2_film + u_acid_lcfa*x_lcfa_film + u_acet_val*x_c4_film + u_acet_but*x_c4_film + u_acet_pro*x_pro_film + u_met_ac*x_ac_film + u_met_h2*x_h2_film - k_det*(x_gl_film + x_aa_film + x_h2_film + x_lcfa_film + x_c4_film + x_c4_film + x_pro_film + x_ac_film + x_h2_film)))
        biogas = ae.gas_flow2(V_liq, rho_ch4, rho_co2, rho_h2, p_tot, p_h2o, temp)
        methane = ae.methane_flow(V_liq, rho_ch4, p_tot, p_h2o, temp)
        NAT = i_n * 14 * 1000
        TA = (2*(i_c / (1+(s_h/ka_co2_2)*(s_h/ka_co2_1)+(s_h/ka_co2_2))) + (i_c/((ka_co2_2/s_h)+(s_h/ka_co2_1)+1)) + ((ac_tot - ((ac_tot*s_h)/(ka_ac+s_h)))/64) + ((pro_tot - ((pro_tot*s_h)/(ka_pro+s_h)))/112) + ((bu_tot - ((bu_tot*s_h)/(ka_bu+s_h)))/160) + ((va_tot - ((va_tot*s_h)/(ka_va+s_h)))/208)) * 100 * 1000


        # Mass balance: volume derivative
        dV_liq = qin - qout
        dV_sol = -qs + ((V_total*rhet_s*(113/160)) / (rho_s*(1-AC_bp/100)*(1-MC_bp/100)))
        deriv_el = (dV_liq*V_total-(dV_liq+dV_sol)*V_liq)/(V_total**2)
        deriv_eg = -(V_gas*(dV_liq+dV_sol))/(V_total**2)
        # Species balance: concentration derivative
        # Chain rule: d(V*Ca)/dt = Ca * dV/dt + V * dCa/dt
        # Composite mass balance
        dx_cdt = (- qout*x_c)/(V_total*el) + lamb*(-x_c*k_dis + (ae.biomass_dead(k_dead, x_h2+(es/el)*x_h2_film)+ae.biomass_dead(k_dead, x_gl+(es/el)*x_gl_film)+ae.biomass_dead(k_dead, x_aa+(es/el)*x_aa_film)+ae.biomass_dead(k_dead, x_lcfa+(es/el)*x_lcfa_film)+ae.biomass_dead(k_dead, x_ac+(es/el)*x_ac_film)+ae.biomass_dead(k_dead, x_pro+(es/el)*x_pro_film)+ae.biomass_dead(k_dead, x_ac+(es/el)*x_ac_film))) - (x_c*(dV_liq+dV_sol)/V_total) - (deriv_el*x_c)/el 
        # Particulate inerts mass balance
        dx_idt = (qin*x_i_in - qout*x_i)/(V_total*el) + lamb*(f_xi_xc*k_dis*x_c) - (x_i*(dV_liq+dV_sol)/V_total) - (deriv_el*x_i)/el 
        # Inerts carbohydrates mass balance
        ds_idt = (qin*s_i_in - qout*s_i)/(V_total*el) + lamb*(f_si_xc*k_dis*x_c) - (s_i*(dV_liq+dV_sol)/V_total) - (deriv_el*s_i)/el 
        # carbohydrates mass balance
        dchdt = (qin*ch_in - qout*ch)/(V_total*el) + lamb*(- (ae.hydrolysis(k_hyd_ch, ch) + f_xch_xc*k_dis*x_c))  - (ch*(dV_liq+dV_sol)/V_total) - (deriv_el*ch)/el
        # protein mass balance
        dprotdt = (qin*pro_in - qout*prot)/(V_total*el) + lamb*(f_xpro_xc*k_dis*x_c - ae.hydrolysis(k_hyd_pro, prot)) - (prot*(dV_liq+dV_sol)/V_total) - (deriv_el*prot)/el
        # lipids mass balance
        dlipdt = (qin*lip_in - qout*lip)/(V_total*el) - lamb*(f_xlip_xc*k_dis*x_c - ae.hydrolysis(k_hyd_lip, lip)) - (lip*(dV_liq+dV_sol)/V_total) - (deriv_el*lip)/el
        # glucose mass balance
        dgldt = (qin*gl_in - qout*gl)/(V_total*el) + lamb*(ae.hydrolysis(k_hyd_ch, ch) + (1-f_lip_lcfa)*ae.hydrolysis(k_hyd_lip, lip) - ((u_acid_su/y_gl)*(x_gl+x_gl_film*(es/el))))  - (gl*(dV_liq+dV_sol)/V_total) - (deriv_el*gl)/el
        # aminoacids mass balance
        daadt = (qin*aa_in - qout*aa)/(V_total*el) + lamb*(ae.hydrolysis(k_hyd_pro, prot) - ((u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el)))) - (aa*(dV_liq+dV_sol)/V_total) - (deriv_el*aa)/el
        # lcfa mass balance
        dlcfadt = (qin*lcfa_in - qout*lcfa)/(V_total*el) + lamb*(f_lip_lcfa*ae.hydrolysis(k_hyd_lip, lip) - ((u_acid_lcfa/y_lcfa)*(x_lcfa+x_lcfa_film*(es/el)))) - (lcfa*(dV_liq+dV_sol)/V_total) - (deriv_el*lcfa)/el
        # total valerate mass balance
        dval_totdt = (qin*va_tot_in - qout*va_tot)/(V_total*el) + lamb*(((1-y_aa)*f_va_aa*(u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el))) - ((u_acet_val/y_c4)*(x_c4+x_c4_film*(es/el)))) - (va_tot*(dV_liq+dV_sol)/V_total) - (deriv_el*va_tot)/el
        # total butyrate mass balance
        dbu_totdt = (qin*bu_tot_in - qout*bu_tot)/(V_total*el) + lamb*(((1-y_gl)*f_bu_gl*(u_acid_su/y_gl)*(x_gl+x_gl_film*(es/el))) + ((1-y_aa)*f_bu_aa*(u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el))) - ((u_acet_but/y_c4)*(x_c4+x_c4_film*(es/el)))) - (bu_tot*(dV_liq+dV_sol)/V_total) - (deriv_el*bu_tot)/el
        # total propionate mass balance
        dpro_totdt = (qin*pro_tot_in - qout*pro_tot)/(V_total*el) + lamb*(((1-y_gl)*f_pro_gl*(u_acid_su/y_gl)*(x_gl+x_gl_film*(es/el))) + ((1-y_aa)*f_pro_aa*(u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el))) + ((1-y_c4)*f_pro_va*(u_acet_val/y_c4)*(x_c4+x_c4_film*(es/el))) - ((u_acet_pro/y_pro)*(x_pro+x_pro_film*(es/el)))) - (pro_tot*(dV_liq+dV_sol)/V_total) - (deriv_el*pro_tot)/el
        # total acetate mass balance
        dac_totdt = (qin*ac_tot_in - qout*ac_tot)/(V_total*el) + lamb*(((1-y_gl)*f_ac_gl*(u_acid_su/y_gl)*(x_gl+x_gl_film*(es/el))) + ((1-y_aa)*f_ac_aa*(u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el))) + ((1-y_lcfa)*f_ac_lcfa*(u_acid_lcfa/y_lcfa)*(x_lcfa+x_lcfa_film*(es/el))) + ((1-y_c4)*f_ac_va*(u_acet_val/y_c4)*(x_c4+x_c4_film*(es/el))) + ((1-y_c4)*f_ac_bu*(u_acet_but/y_c4)*(x_c4+x_c4_film*(es/el))) + ((1-y_pro)*f_ac_pro*(u_acet_pro/y_pro)*(x_pro+x_pro_film*(es/el))) - ((u_met_ac/y_ac)*(x_ac+x_ac_film*(es/el)))) - (ac_tot*(dV_liq+dV_sol)/V_total) - (deriv_el*ac_tot)/el
        # soluble hydrogen mass balance
        dh2_dt = (-qout*h2)/(V_total*el) + lamb*(((1-y_gl)*f_h2_gl*(u_acid_su/y_gl)*(x_gl+x_gl_film*(es/el))) + ((1-y_aa)*f_h2_aa*(u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el))) + ((1-y_lcfa)*f_h2_lcfa*(u_acid_lcfa/y_lcfa)*(x_lcfa+x_lcfa_film*(es/el))) + ((1-y_c4)*f_h2_va*(u_acet_val/y_c4)*(x_c4+x_c4_film*(es/el))) + ((1-y_c4)*f_h2_bu*(u_acet_but/y_c4)*(x_c4+x_c4_film*(es/el))) + ((1-y_pro)*f_h2_pro*(u_acet_pro/y_pro)*(x_pro+x_pro_film*(es/el))) - ((u_met_h2/y_h2)*(x_h2+x_h2_film*(es/el))) - rho_h2) - (h2*(dV_liq+dV_sol)/V_total) - (deriv_el*h2)/el
        # Hydrogen gass mass balance
        dh2_gdt = lamb*(-h2_g*ae.gas_flow2(V_liq, rho_ch4, rho_co2, rho_h2, p_tot, p_h2o, temp) + (rho_h2/16)*V_liq) - (h2_g*(dV_liq+dV_sol)/V_total) - (deriv_eg*h2_g)/eg        
        # liquid methane mass balance
        dch4dt = (- qout*ch4)/(V_total*el) + lamb*(((1-y_h2)*(u_met_h2/y_h2)*(x_h2+x_h2_film*(es/el))) + (1-y_ac)*(u_met_ac/y_ac)*(x_ac+x_ac_film*(es/el)) - rho_ch4) - (ch4*(dV_liq+dV_sol)/V_total) - (deriv_el*ch4)/el
        # Gas methane mass balance
        dch4_gdt = lamb*(-ch4_g*ae.gas_flow2(V_liq, rho_ch4, rho_co2, rho_h2, p_tot, p_h2o, temp) + (rho_ch4/64)*V_liq) - (ch4_g*(dV_liq+dV_sol)/V_total) - (deriv_eg*ch4_g)/eg
        # inorganic carbon mass balance
        di_cdt = (qin*i_c_in - qout*i_c)/(V_total*el) + lamb*( - (-c_gl + (1-y_gl)*f_bu_gl*c_bu + (1-y_gl)*f_pro_gl*c_pro + (1-y_gl)*f_ac_gl*c_ac + (1-y_gl)*f_h2_gl*c_h2 + y_gl*c_bm)*(u_acid_su/y_gl)*(x_gl+x_gl_film*(es/el)) - (-c_aa + (1-y_aa)*f_va_aa*c_va + (1-y_aa)*f_bu_aa*c_bu + (1-y_aa)*f_pro_aa*c_pro + (1-y_aa)*f_ac_aa*c_ac + (1-y_aa)*f_h2_aa*c_h2 + y_aa*c_bm)*(u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el)) - (-c_pro + (1-y_pro)*f_ac_pro*c_ac + (1-y_pro)*f_h2_pro*c_h2 + y_pro*c_bm)*(u_acet_pro/y_pro)*(x_pro+x_pro_film*(es/el)) - (-c_ac + (1-y_ac)*c_ch4 + y_ac*c_bm)*(u_met_ac/y_ac)*(x_ac+x_ac_film*(es/el)) - (-c_h2 + (1-y_h2)*c_ch4 + y_h2*c_bm)*(u_met_h2/y_h2)*(x_h2+x_h2_film*(es/el)) - rho_co2) - (i_c*(dV_liq+dV_sol)/V_total) - (deriv_el*i_c)/el
        #Gas carbon dioxide mass balance
        dco2_gdt = lamb*(-co2_g*ae.gas_flow2(V_liq, rho_ch4, rho_co2, rho_h2, p_tot, p_h2o, temp) + rho_co2*V_liq) - (co2_g*(dV_liq+dV_sol)/V_total) - (deriv_eg*co2_g)/eg
        # inorganic nitrogen mass balance
        di_ndt = (qin*i_n_in - qout*i_n)/(V_total*el) + lamb*( - n_bio*y_gl*(u_acid_su/y_gl)*(x_gl+x_gl_film*(es/el)) + (n_aa-y_aa*n_bio)*(u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el)) - n_bio*y_lcfa*(u_acid_lcfa/y_lcfa)*(x_lcfa+x_lcfa_film*(es/el)) - n_bio*y_c4*(u_acet_val/y_c4)*(x_c4+x_c4_film*(es/el)) - n_bio*y_pro*(u_acet_pro/y_pro)*(x_pro+x_pro_film*(es/el)) - n_bio*y_c4*(u_acet_but/y_c4)*(x_c4+x_c4_film*(es/el)) - n_bio*y_ac*(u_met_ac/y_ac)*(x_ac+x_ac_film*(es/el)) - n_bio*y_h2*(u_met_h2/y_h2)*(x_h2+x_h2_film*(es/el))) - (i_n*(dV_liq+dV_sol)/V_total) - (deriv_el*i_n)/el
        # glucose degraders mass balance
        dx_gl_filmdt = (- qs*x_gl_film)/(V_total*es) + lamb*(u_acid_su*x_gl_film - ae.biomass_dead(k_dead, x_gl_film) - k_det*x_gl_film) - (x_gl_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_gl_film)/es
        # aminoacids degraders mass balance
        dx_aa_filmdt = (- qs*x_aa_film)/(V_total*es) + lamb*(u_acid_aa*x_aa_film - ae.biomass_dead(k_dead, x_aa_film) - k_det*x_aa_film) - (x_aa_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_aa_film)/es
        # lcfa degraders mass balance
        dx_lcfa_filmdt = (- qs*x_lcfa_film)/(V_total*es) + lamb*(u_acid_lcfa*x_lcfa_film - ae.biomass_dead(k_dead, x_lcfa_film) - k_det*x_lcfa_film) - (x_lcfa_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_lcfa_film)/es
        # valerate degraders mass balance
        dx_c4_filmdt = (- qs*x_c4_film)/(V_total*es) + lamb*(u_acet_val*x_c4_film + u_acet_but*x_c4_film - ae.biomass_dead(k_dead, x_c4_film) - k_det*x_c4_film) - (x_c4_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_c4_film)/es
        # propionic degraders mass balance
        dx_pro_filmdt = (- qs*x_pro_film)/(V_total*es) + lamb*(u_acet_pro*x_pro_film - ae.biomass_dead(k_dead, x_pro_film) - k_det*x_pro_film) - (x_pro_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_pro_film)/es
        # acetate degraders mass balance
        dx_ac_filmdt = (- qs*x_ac_film)/(V_total*es) + lamb*(u_met_ac*x_ac_film - ae.biomass_dead(k_dead, x_ac_film) - k_det*x_ac_film) - (x_ac_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_ac_film)/es
        # hydrogen degraders mass balance
        dx_h2_filmdt = (- qs*x_h2_film)/(V_total*es) + lamb*(u_met_h2*x_h2_film - ae.biomass_dead(k_dead, x_h2_film) - k_det*x_h2_film) - (x_h2_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_h2_film)/es
        # glucose degraders mass balance
        dx_gldt = (- qout*x_gl)/(V_total*el) + lamb*(u_acid_su*x_gl - ae.biomass_dead(k_dead, x_gl) + k_det*x_gl_film*(es/el)) - (x_gl*(dV_liq+dV_sol)/V_total) - (deriv_el*x_gl)/el
        # aminoacids degraders mass balance
        dx_aadt = (- qout*x_aa)/(V_total*el) + lamb*(u_acid_aa*x_aa - ae.biomass_dead(k_dead, x_aa) + k_det*x_aa_film*(es/el)) - (x_aa*(dV_liq+dV_sol)/V_total) - (deriv_el*x_aa)/el
        # lcfa degraders mass balance
        dx_lcfadt = (- qout*x_lcfa)/(V_total*el) + lamb*(u_acid_lcfa*x_lcfa - ae.biomass_dead(k_dead, x_lcfa) + k_det*x_lcfa_film*(es/el)) - (x_lcfa*(dV_liq+dV_sol)/V_total) - (deriv_el*x_lcfa)/el
        # valerate degraders mass balance
        dx_c4dt = (- qout*x_c4)/(V_total*el) + lamb*(u_acet_val*x_c4 + u_acet_but*x_c4 - ae.biomass_dead(k_dead, x_c4) + k_det*x_c4_film*(es/el)) - (x_c4*(dV_liq+dV_sol)/V_total) - (deriv_el*x_c4)/el
        # propionic degraders mass balance
        dx_prodt = (- qout*x_pro)/(V_total*el) + lamb*(u_acet_pro*x_pro - ae.biomass_dead(k_dead, x_pro) + k_det*x_pro_film*(es/el)) - (x_pro*(dV_liq+dV_sol)/V_total) - (deriv_el*x_pro)/el
        # acetate degraders mass balance
        dx_acdt = (- qout*x_ac)/(V_total*el) + lamb*(u_met_ac*x_ac - ae.biomass_dead(k_dead, x_ac) + k_det*x_ac_film*(es/el)) - (x_ac*(dV_liq+dV_sol)/V_total) - (deriv_el*x_ac)/el
        # hydrogen degraders mass balance
        dx_h2dt = (- qout*x_h2)/(V_total*el) + lamb*(u_met_h2*x_h2 - ae.biomass_dead(k_dead, x_h2) + k_det*x_h2_film*(es/el)) - (x_h2*(dV_liq+dV_sol)/V_total) - (deriv_el*x_h2)/el
        # Cations mass balance
        dcatdt = (qin*cat_in - qout*cat)/(V_total*el) - (cat*(dV_liq+dV_sol)/V_total) - (deriv_el*cat)/el
        # Anions mass balance
        dandt = (qin*an_in - qout*an)/(V_total*el) - (an*(dV_liq+dV_sol)/V_total) - (deriv_el*an)/el
        # Cumulated methane in CNPT
        dcum_methanedt = (methane/temp)*273.15
        # Cumulated biogas in CNPT
        dcum_biogasdt = (biogas/temp)*273.15

        # average bioparticle diameter
        if lamb == 0:
            dbpdt = 0
        else:
            dbpdt = (dbp/3)*(rhet_s/(es*(x_lcfa_film+x_aa_film+x_c4_film+x_gl_film+x_ac_film+x_pro_film+x_h2_film)))

        f = np.array([dV_liq, dx_cdt, dx_idt, ds_idt, dchdt, dprotdt, dlipdt, dgldt, daadt, dlcfadt, dval_totdt, dbu_totdt, dpro_totdt, dac_totdt, dh2_dt, dh2_gdt, dch4dt, dch4_gdt, di_cdt, dco2_gdt, di_ndt, dx_gldt, dx_aadt, dx_lcfadt, dx_c4dt, dx_prodt, dx_acdt, dx_h2dt, dV_sol, dx_gl_filmdt, dx_aa_filmdt, dx_lcfa_filmdt, dx_c4_filmdt, dx_pro_filmdt, dx_ac_filmdt, dx_h2_filmdt, dbpdt, dcatdt, dandt, dcum_methanedt, dcum_biogasdt, pH])
        
        # Return derivatives
        return f


    def ode_wrapper(self, t, y):
        resultado = self.vessel(t, y)

        dydt = resultado[:-1]
        alg_var = resultado[-1]

        self.algebraic_vars.append(alg_var)
        self.tie.append(t)

        return dydt

    # Function that solves the DAE system
    def ode(self):
        
        # Differential equations
        states = solve_ivp(self.ode_wrapper, self.tspan, self.initial, t_eval=self.teval, method='BDF', rtol=1E-9, atol=1E-12)
        
        time = states.t
        v_liq = states.y[0]
        composites = states.y[1]
        part_inert = states.y[2]
        sol_inert = states.y[3]
        carbohydrates = states.y[4]
        proteins = states.y[5]
        lipids = states.y[6]
        sugars = states.y[7]
        aminoacids = states.y[8]
        lcfa = states.y[9]
        total_val = states.y[10]
        total_bu = states.y[11]
        total_pro = states.y[12]
        total_ac = states.y[13]
        hydrogen = states.y[14]
        hydrogen_gas = states.y[15]
        ch4_liq = states.y[16]
        ch4_gas = states.y[17]
        inorganic_carbon = states.y[18]
        co2_gas = states.y[19]
        inorganic_nitrogen = states.y[20]
        biomass_su = states.y[21]
        biomass_aa = states.y[22]
        biomass_lcfa = states.y[23]
        biomass_c4 = states.y[24]
        biomass_pro = states.y[25]
        biomass_ac = states.y[26]
        biomass_h2 = states.y[27]
        v_sol = states.y[28]
        biomass_su_film = states.y[29]
        biomass_aa_film = states.y[30]
        biomass_lcfa_film = states.y[31]
        biomass_c4_film = states.y[32]
        biomass_pro_film = states.y[33]
        biomass_ac_film = states.y[34]
        biomass_h2_film = states.y[35]
        dbp = states.y[36]
        cat = states.y[37]
        an = states.y[38]
        cum_methane = states.y[39]
        cum_biogas = states.y[40]

        # Algebraic equations
        
        COD_Total = (v_sol/v_liq)*(biomass_su_film + biomass_aa_film + biomass_c4_film + biomass_lcfa_film + biomass_pro_film + biomass_ac_film + biomass_h2_film) + carbohydrates + proteins + lipids + part_inert + sol_inert + sugars + aminoacids + lcfa + total_val + total_bu + total_pro + total_ac + biomass_su + biomass_aa + biomass_lcfa + biomass_c4 + biomass_pro + biomass_ac + biomass_h2

        COD_Soluble = sugars + aminoacids + lcfa + total_val + total_bu + total_pro + total_ac

        S_Sout = sugars + total_ac + total_pro + total_val + total_bu + aminoacids + lcfa

        X_Sout = lipids + proteins + carbohydrates + part_inert + biomass_c4 + biomass_h2 + biomass_lcfa + biomass_pro + biomass_ac + biomass_su + biomass_aa

        S_Iout = sol_inert

        S_NH4out = inorganic_nitrogen*14

        S_Alkout = inorganic_carbon

        i_NSS_An = 0

        i_NXS_An = 0

        f_CH_XS = carbohydrates/X_Sout
        f_P_XS = proteins/X_Sout
        f_L_XS = lipids/X_Sout

        V_total = v_liq + self.cts[37] + v_sol

        return time, v_liq, composites, part_inert, sol_inert, carbohydrates, proteins, lipids, sugars, aminoacids, lcfa, total_val, total_bu, total_pro, total_ac, hydrogen, hydrogen_gas, ch4_liq, ch4_gas, inorganic_carbon, co2_gas, inorganic_nitrogen, biomass_su, biomass_aa, biomass_lcfa, biomass_c4, biomass_pro, biomass_ac, biomass_h2, v_sol, biomass_su_film, biomass_aa_film, biomass_lcfa_film, biomass_c4_film, biomass_pro_film, biomass_ac_film, biomass_h2_film, dbp, cat, an, COD_Total, cum_biogas, cum_methane, COD_Soluble, S_Iout, S_NH4out, S_Alkout, i_NSS_An, i_NXS_An, f_CH_XS, f_P_XS, f_L_XS, V_total, self.algebraic_vars, self.tie
