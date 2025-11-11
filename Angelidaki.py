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
        # Concentration of glucose (gCOD/L)
        gl = x[6]
        # Concentration of aminoacids (gCOD/L) 
        aa = x[7]
        # Concentration of Long Chain Fatty Acids (gCOD/L)
        lcfa = x[8]
        # Concentration of total valerate (gCOD/L)
        va_tot = x[9]
        # Concentration of total butirate (gCOD/L)
        bu_tot = x[10]
        # Concentration of total propionate (gCOD/L)
        pro_tot = x[11]
        # Concentration of total acetate (gCOD/L)
        ac_tot = x[12]
        # Concentration of liquid methane (gCOD/L)
        ch4 = x[13]
        # Concentration of gas methane (gCOD/L)
        ch4_g = x[14]
        # Concentration of inorganic carbon (mol/L)
        i_c = x[15]
        # Concentration of gas carbon dioxide (mol/L)
        co2_g = x[16]
        # Concentration of inorganic nitrogen (mol/L)
        i_n = x[17]
        # Concentration of lipids degraders (gCOD/L)
        x_lip = x[18]
        # Concentration of glucose degraders (gCOD/L)
        x_gl = x[19]
        # Concentration of aminoacids degraders (gCOD/L)
        x_aa = x[20]
        # Concentration of lcfa degraders (gCOD/L) 
        x_lcfa = x[21]
        # Concentration of valerate degraders (gCOD/L)
        x_va = x[22]
        # Concentration of butirate degraders (gCOD/L)
        x_bu = x[23]
        # Concentration of propionate degraders (gCOD/L)
        x_pro = x[24]
        # Concentration of acetate degraders (gCOD/L)
        x_ac = x[25]
        # Solid volume [L]
        V_sol = x[26]
        # Concentration of lipids degraders in solid phase (gCOD/L)
        x_lip_film = x[27]
        # Concentration of glucose degraders in solid phase (gCOD/L)
        x_gl_film = x[28]
        # Concentration of aminoacids degraders in solid phase (gCOD/L)
        x_aa_film = x[29]
        # Concentration of lcfa degraders in solid phase (gCOD/L)
        x_lcfa_film = x[30]
        # Concentration of valerate degraders in solid phase (gCOD/L)
        x_va_film = x[31]
        # Concentration of butyrate degraders in solid phase (gCOD/L)
        x_bu_film = x[32]
        # Concentration of propionate degraders in solid phase (gCOD/L)
        x_pro_film = x[33]
        # Concentration of acetate degraders in solid phase (gCOD/L)
        x_ac_film = x[34]
        # Average bioparticle diameter [dm]
        dbp = x[35]
        # Concentration of cations [mol/L]
        cat = x[36]
        # Concentration of cations [mol/L]
        an = x[37]
        # Cumulative methane [L]
        cum_methane = x[38]
        # Cumulative biogas [L]
        cum_biogas = x[39]
        # Energy produced
        Ep = x[40]
        # Energy consumed by agitation
        Ec_ag = x[41]
        # Energy consumed by calor
        Ec_cal = x[42]
        # Cumulative energy consumed
        Ec = x[43]

        # Inputs (3):
        #  pars  = kinetics parameters (47)
        #   hydrolysis of proteins
        k_hyd_pro = self.pars[0]
        #   hydrolysis of carbohydrates
        k_hyd_ch = self.pars[1]    
        #   death of biomass
        k_dead = self.pars[2]
        #   Ks of inorganic nitrogen
        ks_in = self.pars[3]
        #   Ki of VFA
        ki_vfa = self.pars[4]
        #   Ki of LCFA
        ki_lcfa = self.pars[5]
        #   Ki of acetate on valerate consumption
        ki_ac_va = self.pars[6]
        #   Ki of acetate on butirate consumption
        ki_ac_bu = self.pars[7]
        #   Ki of acetate on propionate consumption
        ki_ac_pro = self.pars[8]
        #   Ki of NH3
        ki_nh3 = self.pars[9]
        #   pH upper limit
        pH_ul = self.pars[10]        
        #   pH lower limit       
        pH_ll = self.pars[11]
        #   Lipolysis
        umax_acid_lip = self.pars[12]
        ks_lip = self.pars[13]
        y_lip = self.pars[14]
        #   Acidogenesis of glucose
        umax_acid_su = self.pars[15]
        ks_gl = self.pars[16]
        y_gl = self.pars[17]    
        #   Acidogenesis of aminoacids
        umax_acid_aa = self.pars[18]
        ks_aa = self.pars[19]
        y_aa = self.pars[20]
        #   Acetogenesis of lcfa
        umax_acid_lcfa = self.pars[21]
        ks_lcfa = self.pars[22]
        y_lcfa = self.pars[23]
        #   Acetogenesis of valerate
        umax_acet_val = self.pars[24]
        ks_va = self.pars[25]
        y_va = self.pars[26]
        #   Acetogenesis of butirate     
        umax_acet_but = self.pars[27]
        ks_bu = self.pars[28]
        y_bu = self.pars[29]
        #   Acetogenesis of propionate
        umax_acet_pro = self.pars[30]
        ks_pro = self.pars[31]
        y_pro = self.pars[32]
        #   Methanogenesis of acetate
        umax_met = self.pars[33]
        ks_ac = self.pars[34]
        y_ac = self.pars[35]
        # Acid base reactions
        deltaH_CH4 = self.pars[36]
        deltaH_CO2 = self.pars[37]
        c_bm = self.pars[38]
        c_ch4 = self.pars[39]
        temp = self.pars[40]
        #   Henry's constants
        kH298_ch4 = self.pars[41]
        kH298_co2 = self.pars[42]
        #   Mass transfer coefficient   
        kLa = self.pars[43]

        #  cts   = Constants (36)
        #   Conversion
        #   inerts of carbohydrates
        f_i_ch = self.cts[0] 
        #   inerts of proteins
        f_i_pro = self.cts[1]
        #   LCFA of lipids
        f_lip_lcfa = self.cts[2]
        #   Propionate of lipids
        f_lip_pro = self.cts[3]
        #   Butirate of glucose
        f_bu_gl = self.cts[4]
        #   Propionate of glucose
        f_pro_gl = self.cts[5]
        #   Acetate of glucose
        f_ac_gl = self.cts[6]
        #   Valerate of aminoacids
        f_va_aa = self.cts[7]
        #   Butirate of aminoacids
        f_bu_aa = self.cts[8]
        #   Propionate of aminoacids
        f_pro_aa = self.cts[9]
        #   Acetate of aminoacids
        f_ac_aa = self.cts[10] 
        #   Acetate of LCFA      
        f_ac_lcfa = self.cts[11]
        #   Methane of LCFA
        f_ch4_lcfa = self.cts[12]
        #   Propionate of valerate
        f_pro_va = self.cts[13]
        #   Acetate of valerate
        f_ac_va = self.cts[14]
        #   Methane of valerate
        f_ch4_va = self.cts[15]
        #   Acetate of butirate
        f_ac_bu = self.cts[16]
        #   Methane of butirate
        f_ch4_bu = self.cts[17]
        #   Acetate of propionate
        f_ac_pro = self.cts[18]
        #   Methane of propionate
        f_ch4_pro = self.cts[19]
        #   Carbohydrate of biomass
        f_bio_ch = self.cts[20]
        #   Protein of biomass 
        f_bio_prot = self.cts[21]

        #   Flows
        #    inlet flow
        qin = self.cts[22]
        #    outlet flow
        qout = self.cts[23]

        #   Nitrogen and carbon conversion
        #   Biomass to nitrogen factor
        n_bio = self.cts[24]
        #   Aminoacids to nitrogen factor
        n_aa = self.cts[25]
        #   lipids to carbon factor
        c_lip = self.cts[26]
        #   glucose to carbon factor
        c_gl = self.cts[27]
        #   aminoacids to carbon factor
        c_aa = self.cts[28]
        #   lcfa to carbon factor
        c_lcfa = self.cts[29]
        #   valerate to carbon factor
        c_va = self.cts[30]
        #   butyrate to carbon factor
        c_bu = self.cts[31]
        #   propionate to carbon factor
        c_pro = self.cts[32]
        #   acetate to carbon factor 
        c_ac = self.cts[33]

        #   Others 
        V_gas = self.cts[34]
        #    Outlet solid flow
        qs = self.cts[35]
        # Mixing constant
        lamb = self.cts[36]
        # Bioparticle-related constants
        rho_s = self.cts[37]
        AC_bp = self.cts[38]
        MC_bp = self.cts[39]
        k_det = self.cts[40]

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
        # Inlet of cations [mol/L]
        cat_in = self.feed[14]
        # Inlet of anions [mol/L]
        an_in = self.feed[15]
        
        
        # Acid base reactions for pH calculation
        ka_nh3 = 10**(-9.25)*math.exp(0.07*(298.15-temp))
        ka_co2_1 = 10**(-6.35)*math.exp(0.01*(298.15-temp))
        ka_co2_2 = 10**(-10.30)*math.exp(0.005*(298.15-temp))
        ka_lcfa = 10**(-4.25)
        ka_va = 10**(-4.86)
        ka_pro = 10**(-4.94)
        ka_bu = 10**(-4.92)
        ka_ac = 10**(-4.81)
        ka_w = 10**(-14)*math.exp(0.076*(298.15-temp))

        def charge_balance(y):
            return cat + ((i_n*y)/(ka_nh3+y)) + y - 2*(i_c/(1+(y/ka_co2_2)*(y/ka_co2_1)+(y/ka_co2_2))) - (i_c/((ka_co2_2/y)+(y/ka_co2_1)+1)) - ((lcfa - ((lcfa*y)/(ka_lcfa+y)))/736) - ((ac_tot - ((ac_tot*y)/(ka_ac+y)))/64) - ((pro_tot - ((pro_tot*y)/(ka_pro+y)))/112) - ((bu_tot - ((bu_tot*y)/(ka_bu+y)))/160) - ((va_tot - ((va_tot*y)/(ka_va+y)))/208) - (ka_w/y) - an        


        s_h = float(fsolve(charge_balance, 1E-7))
        #print(s_h)
        co2_liq = (i_c/((ka_co2_2/s_h)+(s_h/ka_co2_1)+1))*(s_h/ka_co2_1)
                                                                             
        
        pH = ae.pH1(s_h)
        
        
        nh3 = (i_n - ((i_n*s_h)/(ka_nh3+s_h)))*17.031
        S_hac = (s_h*ac_tot)/(ka_ac+s_h)

        # Pressures
        p_ch4 = ae.pres(ch4_g, temp)
        p_co2 = ae.pres(co2_g, temp)
        p_h2o = 0.0313*math.exp(5290*((1/298.15)-(1/temp)))
        p_tot = 1.013


        # Transfer terms
        kH_ch4 = ae.kH_temp(kH298_ch4, deltaH_CH4, temp)
        kH_co2 = ae.kH_temp(kH298_co2, deltaH_CO2, temp)
        rho_ch4 = ae.rho_gas(lamb, kLa, ch4, kH_ch4, p_ch4*64)
        rho_co2 = ae.rho_gas(lamb, kLa, co2_liq, kH_co2, p_co2) 


        # Temperature dependence
        umax_acid_su_Te = umax_acid_su #- 0.0936*(55 - (temp-273.15))
        umax_acid_aa_Te = umax_acid_aa #- 0.1176*(55 - (temp-273.15))
        umax_acid_lip_Te = umax_acid_lip #- 0.00984*(55 - (temp-273.15))
        umax_acid_lcfa_Te = umax_acid_lcfa #- 0.01008*(55 - (temp-273.15))
        umax_acet_pro_Te = umax_acet_pro #- 0.00888*(53 - (temp-273.15))
        umax_acet_but_Te = umax_acet_but #- 0.01248*(60 - (temp-273.15))
        umax_acet_val_Te = umax_acet_val #- 0.01272*(60 - (temp-273.15))
        umax_met_Te = umax_met #- 0.01128*(55 - (temp-273.15))

        # Growth rate equations
        u_acid_su = umax_acid_su_Te * (1/(1+(ks_gl/gl))) * ae.inhibition2(ks_in, i_n) * ae.inhibition1(ki_lcfa, lcfa)
        u_acid_aa = umax_acid_aa_Te * (1/(1+(ks_aa/aa)))
        u_acid_lip = umax_acid_lip_Te * (1/(1+(ks_lip/lip))) * ae.inhibition2(ks_in, i_n) * ae.inhibition1(ki_lcfa, lcfa) * ae.inhibitionpH(pH_ll, pH_ul, pH)
        u_acid_lcfa = umax_acid_lcfa_Te * (1/(1+(ks_lcfa/lcfa)+(lcfa/ki_lcfa))) * ae.inhibition2(ks_in, i_n) * ae.inhibitionpH(pH_ll, pH_ul, pH)
        u_acet_val = umax_acet_val_Te * (1/(1+(ks_va/va_tot))) * ae.inhibition2(ks_in, i_n) * ae.inhibition1(ki_lcfa, lcfa) * (1/(1+(S_hac/ki_ac_va))) * ae.inhibitionpH(pH_ll, pH_ul, pH)
        u_acet_pro = umax_acet_pro_Te * (1/(1+(ks_pro/pro_tot))) * ae.inhibition2(ks_in, i_n) * ae.inhibition1(ki_lcfa, lcfa) * (1/(1+(S_hac/ki_ac_pro))) * ae.inhibitionpH(pH_ll, pH_ul, pH)
        u_acet_but = umax_acet_but_Te * (1/(1+(ks_bu/bu_tot))) * ae.inhibition2(ks_in, i_n) * ae.inhibition1(ki_lcfa, lcfa) * (1/(1+(S_hac/ki_ac_bu))) * ae.inhibitionpH(pH_ll, pH_ul, pH)
        u_met = umax_met_Te * (1/(1+(ks_ac/ac_tot))) * ae.inhibition2(ks_in, i_n) * ae.inhibition1(ki_lcfa, lcfa) * ae.inhibition1(ki_nh3, nh3) * ae.inhibitionpH(pH_ll, pH_ul, pH)

        lcfa_inhibition = ae.inhibition1(ki_lcfa, lcfa)
        

        # Other algebraic equations
        V_total = V_liq + V_gas + V_sol
        el = V_liq/V_total
        es = V_sol/V_total
        eg = V_gas/V_total
        rhet_s = lamb*(es*(u_acid_su*x_gl_film + u_acid_aa*x_aa_film + u_acid_lip*x_lip_film + u_acid_lcfa*x_lcfa_film + u_acet_val*x_va_film + u_acet_but*x_bu_film + u_acet_pro*x_pro_film + u_met*x_ac_film - k_det*(x_gl_film + x_aa_film + x_lip_film + x_lcfa_film + x_va_film + x_bu_film + x_pro_film + x_ac_film)))
        biogas = ae.gas_flow(V_liq, rho_ch4, rho_co2, p_tot, p_h2o, temp)
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
        # Inerts protein mass balance
        dpro_idt = (qin*pro_i_in - qout*pro_i)/(V_total*el) + lamb*(f_i_pro*ae.hydrolysis(k_hyd_pro, prot)*ae.inhibition1(ki_vfa, S_hac)) - (pro_i*(dV_liq+dV_sol)/V_total) - (deriv_el*pro_i)/el 
        # Inerts carbohydrates mass balance
        dch_idt = (qin*ch_i_in - qout*ch_i)/(V_total*el) + lamb*(f_i_ch*ae.hydrolysis(k_hyd_ch, ch)*ae.inhibition1(ki_vfa, S_hac)) - (ch_i*(dV_liq+dV_sol)/V_total) - (deriv_el*ch_i)/el 
        # carbohydrates mass balance
        dchdt = (qin*ch_in - qout*ch)/(V_total*el) + lamb*(- (ae.hydrolysis(k_hyd_ch, ch)*ae.inhibition1(ki_vfa, S_hac) + f_bio_ch*(ae.biomass_dead(k_dead, x_lip)+ae.biomass_dead(k_dead, x_gl)+ae.biomass_dead(k_dead, x_aa)+ae.biomass_dead(k_dead, x_lcfa)+ae.biomass_dead(k_dead, x_va)+ae.biomass_dead(k_dead, x_bu)+ae.biomass_dead(k_dead, x_pro)+ae.biomass_dead(k_dead, x_ac))))  - (ch*(dV_liq+dV_sol)/V_total) - (deriv_el*ch)/el
        # protein mass balance
        dprotdt = (qin*pro_in - qout*prot)/(V_total*el) + lamb*(f_bio_prot*(ae.biomass_dead(k_dead, x_lip)+ae.biomass_dead(k_dead, x_gl)+ae.biomass_dead(k_dead, x_aa)+ae.biomass_dead(k_dead, x_lcfa)+ae.biomass_dead(k_dead, x_va)+ae.biomass_dead(k_dead, x_bu)+ae.biomass_dead(k_dead, x_pro)+ae.biomass_dead(k_dead, x_ac)) - ae.hydrolysis(k_hyd_pro, prot)*ae.inhibition1(ki_vfa, S_hac)) - (prot*(dV_liq+dV_sol)/V_total) - (deriv_el*prot)/el
        # lipids mass balance
        dlipdt = (qin*lip_in - qout*lip)/(V_total*el) - lamb*((u_acid_lip/y_lip)*(x_lip+x_lip_film*(es/el))) - (lip*(dV_liq+dV_sol)/V_total) - (deriv_el*lip)/el
        # glucose mass balance
        dgldt = (qin*gl_in - qout*gl)/(V_total*el) + lamb*((1-f_i_ch)*ae.hydrolysis(k_hyd_ch, ch)*ae.inhibition1(ki_vfa, S_hac) - ((u_acid_su/y_gl)*(x_gl+x_gl_film*(es/el))))  - (gl*(dV_liq+dV_sol)/V_total) - (deriv_el*gl)/el
        # aminoacids mass balance
        daadt = (qin*aa_in - qout*aa)/(V_total*el) + lamb*((1-f_i_pro)*ae.hydrolysis(k_hyd_pro, prot)*ae.inhibition1(ki_vfa, S_hac) - ((u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el)))) - (aa*(dV_liq+dV_sol)/V_total) - (deriv_el*aa)/el
        # lcfa mass balance
        dlcfadt = (qin*lcfa_in - qout*lcfa)/(V_total*el) + lamb*(((1-y_lip)*f_lip_lcfa*(u_acid_lip/y_lip)*(x_lip+x_lip_film*(es/el))) - ((u_acid_lcfa/y_lcfa)*(x_lcfa+x_lcfa_film*(es/el)))) - (lcfa*(dV_liq+dV_sol)/V_total) - (deriv_el*lcfa)/el
        # total valerate mass balance
        dval_totdt = (qin*va_tot_in - qout*va_tot)/(V_total*el) + lamb*(((1-y_aa)*f_va_aa*(u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el))) - ((u_acet_val/y_va)*(x_va+x_va_film*(es/el)))) - (va_tot*(dV_liq+dV_sol)/V_total) - (deriv_el*va_tot)/el
        # total butyrate mass balance
        dbu_totdt = (qin*bu_tot_in - qout*bu_tot)/(V_total*el) + lamb*(((1-y_gl)*f_bu_gl*(u_acid_su/y_gl)*(x_gl+x_gl_film*(es/el))) + ((1-y_aa)*f_bu_aa*(u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el))) - ((u_acet_but/y_bu)*(x_bu+x_bu_film*(es/el)))) - (bu_tot*(dV_liq+dV_sol)/V_total) - (deriv_el*bu_tot)/el
        # total propionate mass balance
        dpro_totdt = (qin*pro_tot_in - qout*pro_tot)/(V_total*el) + lamb*(((1-y_lip)*f_lip_pro*(u_acid_lip/y_lip)*(x_lip+x_lip_film*(es/el))) + ((1-y_gl)*f_pro_gl*(u_acid_su/y_gl)*(x_gl+x_gl_film*(es/el))) + ((1-y_aa)*f_pro_aa*(u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el))) + ((1-y_va)*f_pro_va*(u_acet_val/y_va)*(x_va+x_va_film*(es/el))) - ((u_acet_pro/y_pro)*(x_pro+x_pro_film*(es/el)))) - (pro_tot*(dV_liq+dV_sol)/V_total) - (deriv_el*pro_tot)/el
        # total acetate mass balance
        dac_totdt = (qin*ac_tot_in - qout*ac_tot)/(V_total*el) + lamb*(((1-y_gl)*f_ac_gl*(u_acid_su/y_gl)*(x_gl+x_gl_film*(es/el))) + ((1-y_aa)*f_ac_aa*(u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el))) + ((1-y_lcfa)*f_ac_lcfa*(u_acid_lcfa/y_lcfa)*(x_lcfa+x_lcfa_film*(es/el))) + ((1-y_va)*f_ac_va*(u_acet_val/y_va)*(x_va+x_va_film*(es/el))) + ((1-y_bu)*f_ac_bu*(u_acet_but/y_bu)*(x_bu+x_bu_film*(es/el))) + ((1-y_pro)*f_ac_pro*(u_acet_pro/y_pro)*(x_pro+x_pro_film*(es/el))) - ((u_met/y_ac)*(x_ac+x_ac_film*(es/el)))) - (ac_tot*(dV_liq+dV_sol)/V_total) - (deriv_el*ac_tot)/el
        # liquid methane mass balance
        dch4dt = (- qout*ch4)/(V_total*el) + lamb*(((1-y_lcfa)*f_ch4_lcfa*(u_acid_lcfa/y_lcfa)*(x_lcfa+x_lcfa_film*(es/el))) + ((1-y_va)*f_ch4_va*(u_acet_val/y_va)*(x_va+x_va_film*(es/el))) + ((1-y_bu)*f_ch4_bu*(u_acet_but/y_bu)*(x_bu+x_bu_film*(es/el))) + ((1-y_pro)*f_ch4_pro*(u_acet_pro/y_pro)*(x_pro+x_pro_film*(es/el))) + (1-y_ac)*(u_met/y_ac)*(x_ac+x_ac_film*(es/el)) - rho_ch4) - (ch4*(dV_liq+dV_sol)/V_total) - (deriv_el*ch4)/el
        # Gas methane mass balance
        dch4_gdt = lamb*(-ch4_g*biogas + (rho_ch4/64)*V_liq) - (ch4_g*(dV_liq+dV_sol)/V_total) - (deriv_eg*ch4_g)/eg
        # inorganic carbon mass balance
        di_cdt = (qin*i_c_in - qout*i_c)/(V_total*el) + lamb*( - ((1-y_lip)*f_lip_lcfa*c_lcfa + (1-y_lip)*f_lip_pro*c_pro - c_lip + y_lip*c_bm )*(u_acid_lip/y_lip)*(x_lip+x_lip_film*(es/el)) - (-c_gl + (1-y_gl)*f_bu_gl*c_bu + (1-y_gl)*f_pro_gl*c_pro + (1-y_gl)*f_ac_gl*c_ac + y_gl*c_bm)*(u_acid_su/y_gl)*(x_gl+x_gl_film*(es/el)) - (-c_aa + (1-y_aa)*f_va_aa*c_va + (1-y_aa)*f_bu_aa*c_bu + (1-y_aa)*f_pro_aa*c_pro + (1-y_aa)*f_ac_aa*c_ac + y_aa*c_bm)*(u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el)) - (-c_lcfa + (1-y_lcfa)*f_ac_lcfa*c_ac + (1-y_lcfa)*f_ch4_lcfa*c_ch4 + y_lcfa*c_bm)*(u_acid_lcfa/y_lcfa)*(x_lcfa+x_lcfa_film*(es/el)) - (-c_va + (1-y_va)*f_pro_va*c_pro + (1-y_va)*f_ac_va*c_ac + (1-y_va)*f_ch4_va*c_ch4 + y_va*c_bm)*(u_acet_val/y_va)*(x_va+x_va_film*(es/el)) - (-c_bu + (1-y_bu)*f_ac_bu*c_ac + (1-y_bu)*f_ch4_bu*c_ch4 + y_bu*c_bm)*(u_acet_but/y_bu)*(x_bu+x_bu_film*(es/el)) - (-c_pro + (1-y_pro)*f_ac_pro*c_ac + (1-y_pro)*f_ch4_pro*c_ch4 + y_pro*c_bm)*(u_acet_pro/y_pro)*(x_pro+x_pro_film*(es/el)) - (-c_ac + (1-y_ac)*c_ch4 + y_ac*c_bm)*(u_met/y_ac)*(x_ac+x_ac_film*(es/el)) - rho_co2) - (i_c*(dV_liq+dV_sol)/V_total) - (deriv_el*i_c)/el
        #Gas carbon dioxide mass balance
        dco2_gdt = lamb*(-co2_g*biogas + rho_co2*V_liq) - (co2_g*(dV_liq+dV_sol)/V_total) - (deriv_eg*co2_g)/eg
        # inorganic nitrogen mass balance
        di_ndt = (qin*i_n_in - qout*i_n)/(V_total*el) + lamb*( - n_bio*y_gl*(u_acid_su/y_gl)*(x_gl+x_gl_film*(es/el)) + (n_aa-y_aa*n_bio)*(u_acid_aa/y_aa)*(x_aa+x_aa_film*(es/el)) - n_bio*y_lcfa*(u_acid_lcfa/y_lcfa)*(x_lcfa+x_lcfa_film*(es/el)) - n_bio*y_va*(u_acet_val/y_va)*(x_va+x_va_film*(es/el)) - n_bio*y_pro*(u_acet_pro/y_pro)*(x_pro+x_pro_film*(es/el)) - n_bio*y_bu*(u_acet_but/y_bu)*(x_bu+x_bu_film*(es/el)) - n_bio*y_ac*(u_met/y_ac)*(x_ac+x_ac_film*(es/el))) - (i_n*(dV_liq+dV_sol)/V_total) - (deriv_el*i_n)/el
        # lipids degraders mass balance
        dx_lip_filmdt = (- qs*x_lip_film)/(V_total*es) + lamb*(u_acid_lip*x_lip_film - ae.biomass_dead(k_dead, x_lip_film) - k_det*x_lip_film) - (x_lip_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_lip_film)/es
        # glucose degraders mass balance
        dx_gl_filmdt = (- qs*x_gl_film)/(V_total*es) + lamb*(u_acid_su*x_gl_film - ae.biomass_dead(k_dead, x_gl_film) - k_det*x_gl_film) - (x_gl_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_gl_film)/es
        # aminoacids degraders mass balance
        dx_aa_filmdt = (- qs*x_aa_film)/(V_total*es) + lamb*(u_acid_aa*x_aa_film - ae.biomass_dead(k_dead, x_aa_film) - k_det*x_aa_film) - (x_aa_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_aa_film)/es
        # lcfa degraders mass balance
        dx_lcfa_filmdt = (- qs*x_lcfa_film)/(V_total*es) + lamb*(u_acid_lcfa*x_lcfa_film - ae.biomass_dead(k_dead, x_lcfa_film) - k_det*x_lcfa_film) - (x_lcfa_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_lcfa_film)/es
        # valerate degraders mass balance
        dx_va_filmdt = (- qs*x_va_film)/(V_total*es) + lamb*(u_acet_val*x_va_film - ae.biomass_dead(k_dead, x_va_film) - k_det*x_va_film) - (x_va_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_va_film)/es
        # butyrate degraders mass balance
        dx_bu_filmdt = (- qs*x_bu_film)/(V_total*es) + lamb*(u_acet_but*x_bu_film - ae.biomass_dead(k_dead, x_bu_film) - k_det*x_bu_film) - (x_bu_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_bu_film)/es
        # propionic degraders mass balance
        dx_pro_filmdt = (- qs*x_pro_film)/(V_total*es) + lamb*(u_acet_pro*x_pro_film - ae.biomass_dead(k_dead, x_pro_film) - k_det*x_pro_film) - (x_pro_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_pro_film)/es
        # acetate degraders mass balance
        dx_ac_filmdt = (- qs*x_ac_film)/(V_total*es) + lamb*(u_met*x_ac_film - ae.biomass_dead(k_dead, x_ac_film) - k_det*x_ac_film) - (x_ac_film*(dV_liq+dV_sol)/V_total) + ((deriv_el+deriv_eg)*x_ac_film)/es
        # lipids degraders mass balance
        dx_lipdt = (- qout*x_lip)/(V_total*el) + lamb*(u_acid_lip*x_lip - ae.biomass_dead(k_dead, x_lip) + k_det*x_lip_film*(es/el)) - (x_lip*(dV_liq+dV_sol)/V_total) - (deriv_el*x_lip)/el
        # glucose degraders mass balance
        dx_gldt = (- qout*x_gl)/(V_total*el) + lamb*(u_acid_su*x_gl - ae.biomass_dead(k_dead, x_gl) + k_det*x_gl_film*(es/el)) - (x_gl*(dV_liq+dV_sol)/V_total) - (deriv_el*x_gl)/el
        # aminoacids degraders mass balance
        dx_aadt = (- qout*x_aa)/(V_total*el) + lamb*(u_acid_aa*x_aa - ae.biomass_dead(k_dead, x_aa) + k_det*x_aa_film*(es/el)) - (x_aa*(dV_liq+dV_sol)/V_total) - (deriv_el*x_aa)/el
        # lcfa degraders mass balance
        dx_lcfadt = (- qout*x_lcfa)/(V_total*el) + lamb*(u_acid_lcfa*x_lcfa - ae.biomass_dead(k_dead, x_lcfa) + k_det*x_lcfa_film*(es/el)) - (x_lcfa*(dV_liq+dV_sol)/V_total) - (deriv_el*x_lcfa)/el
        # valerate degraders mass balance
        dx_vadt = (- qout*x_va)/(V_total*el) + lamb*(u_acet_val*x_va - ae.biomass_dead(k_dead, x_va) + k_det*x_va_film*(es/el)) - (x_va*(dV_liq+dV_sol)/V_total) - (deriv_el*x_va)/el
        # butyrate degraders mass balance
        dx_budt = (- qout*x_bu)/(V_total*el) + lamb*(u_acet_but*x_bu - ae.biomass_dead(k_dead, x_bu) + k_det*x_bu_film*(es/el)) - (x_bu*(dV_liq+dV_sol)/V_total) - (deriv_el*x_bu)/el
        # propionic degraders mass balance
        dx_prodt = (- qout*x_pro)/(V_total*el) + lamb*(u_acet_pro*x_pro - ae.biomass_dead(k_dead, x_pro) + k_det*x_pro_film*(es/el)) - (x_pro*(dV_liq+dV_sol)/V_total) - (deriv_el*x_pro)/el
        # acetate degraders mass balance
        dx_acdt = (- qout*x_ac)/(V_total*el) + lamb*(u_met*x_ac - ae.biomass_dead(k_dead, x_ac) + k_det*x_ac_film*(es/el)) - (x_ac*(dV_liq+dV_sol)/V_total) - (deriv_el*x_ac)/el
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
            dbpdt = (dbp/3)*(rhet_s/(es*(x_lip_film+x_lcfa_film+x_aa_film+x_va_film+x_gl_film+x_ac_film+x_pro_film+x_bu_film)))

        # Energy produced
        dEpdt = (methane/temp)*273.15*10

        # Energy consumed by agitation
        dEc_agdt = lamb*0.08918826448*24

        # Energy consumed by calor
        dEc_caldt = lamb*(25*4.186*0.3859*2.5)/3600*24

        # Cumulative energy consumed
        dEcdt = dEc_agdt + dEc_caldt
        
        f = np.array([dV_liq, dch_idt, dpro_idt, dchdt, dprotdt, dlipdt, dgldt, daadt, dlcfadt, dval_totdt, dbu_totdt, dpro_totdt, dac_totdt, dch4dt, dch4_gdt, di_cdt, dco2_gdt, di_ndt, dx_lipdt, dx_gldt, dx_aadt, dx_lcfadt, dx_vadt, dx_budt, dx_prodt, dx_acdt, dV_sol, dx_lip_filmdt, dx_gl_filmdt, dx_aa_filmdt, dx_lcfa_filmdt, dx_va_filmdt, dx_bu_filmdt, dx_pro_filmdt, dx_ac_filmdt, dbpdt, dcatdt, dandt, dcum_methanedt, dcum_biogasdt, dEpdt, dEc_agdt, dEc_caldt, dEcdt, pH])
        
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
        ch_inert = states.y[1]
        prot_inert = states.y[2]
        carbohydrates = states.y[3]
        proteins = states.y[4]
        lipids = states.y[5]
        sugars = states.y[6]
        aminoacids = states.y[7]
        lcfa = states.y[8]
        total_val = states.y[9]
        total_bu = states.y[10]
        total_pro = states.y[11]
        total_ac = states.y[12]
        ch4_liq = states.y[13]
        ch4_gas = states.y[14]
        inorganic_carbon = states.y[15]
        co2_gas = states.y[16]
        inorganic_nitrogen = states.y[17]
        biomass_lip = states.y[18]
        biomass_su = states.y[19]
        biomass_aa = states.y[20]
        biomass_lcfa = states.y[21]
        biomass_va = states.y[22]
        biomass_bu = states.y[23]
        biomass_pro = states.y[24]
        biomass_ac = states.y[25]
        v_sol = states.y[26]
        biomass_lip_film = states.y[27]
        biomass_su_film = states.y[28]
        biomass_aa_film = states.y[29]
        biomass_lcfa_film = states.y[30]
        biomass_va_film = states.y[31]
        biomass_bu_film = states.y[32]
        biomass_pro_film = states.y[33]
        biomass_ac_film = states.y[34]
        dbp = states.y[35]
        cat = states.y[36]
        an = states.y[37]
        cum_methane = states.y[38]
        cum_biogas = states.y[39]
        Ep = states.y[40]
        Ec_agit = states.y[41]
        Ec_cal = states.y[42]
        Ec = states.y[43]

        # Algebraic equations
        
        COD_Total = (v_sol/v_liq)*(biomass_su_film + biomass_aa_film + biomass_va_film + biomass_bu_film + biomass_lcfa_film + biomass_pro_film + biomass_ac_film + biomass_lip_film) + carbohydrates + proteins + lipids + prot_inert + ch_inert + sugars + aminoacids + lcfa + total_val + total_bu + total_pro + total_ac + biomass_su + biomass_aa + biomass_lip + biomass_bu + biomass_va + biomass_lcfa + biomass_pro + biomass_ac

        COD_Soluble = sugars + aminoacids + lcfa + total_val + total_bu + total_pro + total_ac

        V_total = v_liq + self.cts[34] + v_sol

        return time, v_liq, ch_inert, prot_inert, carbohydrates, proteins, lipids, sugars, aminoacids, lcfa, total_val, total_bu, total_pro, total_ac, ch4_liq, ch4_gas, inorganic_carbon, co2_gas, inorganic_nitrogen, biomass_lip, biomass_su, biomass_aa, biomass_lcfa, biomass_va, biomass_bu, biomass_pro, biomass_ac, v_sol, biomass_lip_film, biomass_su_film, biomass_aa_film, biomass_lcfa_film, biomass_va_film, biomass_bu_film, biomass_pro_film, biomass_ac_film, dbp, cat, an, COD_Total, cum_methane, cum_biogas, COD_Soluble, V_total, self.algebraic_vars, self.tie, Ep, Ec_agit, Ec_cal, Ec
