import Angelidaki as an
import Substrate_tank as st
import AnaerobicEfluent_tank as anef
import ASM as asm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])


# Initial Conditions for the States [26]
#[V_liq0, ch_i0, pro_i0, ch0, pro0, lip0, i_c0, i_n0, gl0, aa0, lcfa0, va_tot0, bu_tot0, pro_tot0, ac_tot0, cat0, an0]
y0_TS = [10000, 0.09, 0.09, 0.051642574, 0.27415651, 1.434984812, 0.4/100, 0.0629333/14, 0.109214796, 0.368394, 2.711362661, 0, 0.28707188, 0.21771537, 0.10664253, 5E-3, 0]
#[V_liq0, ch_i0, pro_i0, ch0, pro0, lip0, gl0, aa0, lcfa0, va_tot0, bu_tot0, pro_tot0, ac_tot0, ch40, ch4_g0, i_c0, co2_gas0, i_n0, x_lip0, x_gl0, x_aa0, x_lcfa0, x_va0, x_bu0, x_pro0, x_ac0, Ta]
y0_TD = [np.float64(2.250000000005076), np.float64(0.11521605646713286), np.float64(0.14593014883079844), np.float64(-9.417697423273619e-05), np.float64(0.00044911668579757395), np.float64(-1.6620187501739211e-21), np.float64(4.865824962061554e-05), np.float64(0.011189906205689973), np.float64(2.7774011003283694), np.float64(0.005070493427844947), np.float64(0.0017625288375789508), np.float64(0.0189409637258213), np.float64(0.024061115908413366), np.float64(0.058410944475597824), 1e-12, np.float64(0.013421296578511712), 1e-06, np.float64(0.006849463827485383), np.float64(0.011055782818550913), np.float64(0.01636028904404511), np.float64(0.03500089453045301), np.float64(0.07980463732720627), np.float64(0.008346411838623453), np.float64(0.02025613104707654), np.float64(0.01583655174956245), np.float64(0.0957931969438174), np.float64(0.6506021802777885), np.float64(1.334572263134382), np.float64(1.9748929888128237), np.float64(4.22504889884029), np.float64(9.633425076172289), np.float64(1.0075170540958565), np.float64(2.4451701970096673), np.float64(1.9116712995050238), np.float64(11.56344564025625), np.float64(0.0248344782276686), np.float64(0.004993581627497684), np.float64(-1.76554215099877e-29), 1e-12, 1e-12, np.float64(192.6401493830928), np.float64(16.41064066432099), np.float64(5.160233361112111), np.float64(21.570874025432023)]
#[V_liq0, Ss0, alk0, o20, nh40, no20, no30, x_S_film0, x_I_film0, x_H_film0, x_A_film0, x_N_film0, x_sto_film0, x_ana_film0, V_sol0, dbp0]
y0_TA = [np.float64(1.7500000000014282), np.float64(0.0003716561264720733), np.float64(0.026276976755228984), np.float64(-1.1566707573805096e-31), np.float64(0.052471575184953184), np.float64(3.475723356824551e-31), np.float64(2.053642278874028e-30), np.float64(40.551701118367504), np.float64(1.5026567628552152), np.float64(4.981592379999658), np.float64(0.02870414258932083), np.float64(0.021999256874299374), np.float64(1.6350028599205297), np.float64(0.9510363650591999), np.float64(0.05012824174393402), np.float64(3.1884804551610095), np.float64(2.0661666666676783), np.float64(5.2546471218276665), np.float64(0.35)]
#[V_liq0, ch_i0, pro_i0, ch0, pro0, lip0, i_c0, i_n0, gl0, aa0, lcfa0, va_tot0, bu_tot0, pro_tot0, ac_tot0, x_lip0, x_gl0, x_aa0, x_lcfa0, x_va0, x_bu0, x_pro0, x_ac0]
y0_TEA = [2.759999999999961, 0.11579163919050892, 0.14488133044143234, -2.6811652417626704e-06, 3.2058079024495623e-05, 7.178911368598348e-17, 0.014597516910293469, 0.007816015876569611, 2.2656810873268838e-05, 0.009652463606502001, 2.367788910815572, 0.00239195457582596, 0.0010262280227860943, 0.009128293066541332, 0.02211906555298736, 0.0006383225583682752, 0.0008418559008833793, 0.0015559066873291234, 0.003893172433329324, 0.0004826870953934151, 0.0009373045881910272, 0.0007721047245260055, 0.004693398866255224]

# Constants
# [qby, qout]
ctes_TS = [194, 264]
# [f_i_ch, f_i_pro, f_lip_lcfa, f_lip_pro, f_bu_gl, f_pro_gl, f_ac_gl, f_va_aa, f_bu_aa, f_pro_aa, f_ac_aa, f_ac_lcfa, f_ch4_lcfa, f_pro_va, f_ac_va, f_ch4_va, f_ac_bu, f_ch4_bu, f_ac_pro, f_ch4_pro, f_ch4_ac, f_bio_ch, f_bio_prot, qin, qout, n_bio, n_aa, c_lip, c_gl, c_aa, c_lcfa, c_va, c_bu, c_pro_ c_ac, V_gas, qin_est]
ctes_TD = [0.5, 0.2, 0.95, 0.05, 0.41, 0.32, 0.27, 0.1, 0.13, 0.12, 0.65, 0.71, 0.29, 0.57, 0.29, 0.14, 0.81, 0.19, 0.59, 0.41, 0.18, 0.82, 0, 0, 0.00625, 0.007, 0.0220, 0.0313, 0.032, 0.0217, 0.0240, 0.0250, 0.0268, 0.0313, 0.01, 0, 0, 1026, 30, 87.5, 0.035]
# [qby, qan, qout, V_gas, qs, lamb, rho_s, AC_bp, MC_bp, k_det, o2s, f_X, f_N_I, f_N_B, f_S_I, f_N_Ss]
ctes_TA = [0, 0, 0, 0.01, 0, 0, 1026, 30, 87.5, 2.75, 0.009, 0.2, 0.04, 0.08, 0.05, 0.4]
# [qin, qout]
ctes_TEA = [0, 194]

# Kinetics parameters
# [k_hyd_pro, k_hyd_ch, k_dead, ks_in, ki_vfa, ki_lcfa, ki_ac_va, ki_ac_bu, ki_ac_pro, ki_nh3, pH_ul, pH_ll, umax_acid_lip, ks_lip, y_lip, umax_acid_su, ks_gl, y_gl, umax_acid_aa, ks_aa, y_aa, umax_acid_lcfa, ks_lcfa, y_lcfa, umax_acid_val, ks_va, y_va, umax_acid_but, ks_bu, y_bu, umax_acid_pro, ks_pro, y_pro, umax_met, ks_ac, y_ac, deltaH_CH4, deltaH_CO2, temp, kH298_CH4, kH298_CO2, kLa]
pars_TD = [10, 10, 0.02, 1E-4, 0.35, 2.5, 0.1427, 0.768, 1.024, 0.0018, 8.5, 6, 0.14, 0.028, 0.0028, 3, 0.5, 0.1, 0.2, 0.3, 0.08, 0.1, 0.4, 0.06, 1.2, 0.3, 0.06, 1.2, 0.3, 0.06, 0.52, 0.3, 0.04, 0.4, 0.15, 0.05, -14240, -19410, 0.0313, 0.0156, 308.15, 0.00142, 0.03545, 200] #44
# [b_A, b_N, b_H, b_sto, ks_X, n_g, k_alk, kLa, ks_sto, ks_ss, k_nh4, kA_no2, kA_no3, kN_no2, kN_no3, kh_no2, kh_no3, k_A_o2, k_N_o2, k_H_o2, umax_A, umax_N, umax_H, k_sto, y_A, y_N, y_H, y_hsto, y_sto, k_h_s, ki_no2]
pars_TA = [0.18, 0.16, 0.26, 0.26, 2.45, 0.6, 0.006, 0, 1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0005, 0.0005, 0.00074, 0.00175, 0.0002, 1.1, 0.96, 9.2, 18.5, 0.18, 0.08, 0.57, 0.68, 0.8, 3, 0.25] #31


# Feed concentrations
# [alk_an, nh4_an, x_S_an, Ss_an, alk_by, nh4_by, x_S_by, Ss_by, o2_in, no2_in, no3_in, i_NSS_by, i_NXS_by]
feed_TA = [0.013788587857412226, 0.09344069712539413, 0.27559774351060096, 2.503796470924258, 0.004, 0.0629333, 1.9407838960000001, 3.800401237, 0.004, 1e-12, 1e-12]
# [ch_i_in_hidro pro_i_in_hidro, ch_in_hidro, pro_in_hidro, lip_in_hidro, gl_in, aa_in, lcfa_in, total_val_in, total_bu_in, total_pro_in, total_ac_in, i_c_in_hidro, i_n_in_hidro, X_lip_in, X_su_in, X_aa_in, X_lcfa_in, ch_i_in_est, pro_i_in_est, ch_in_est, pro_in_est, lip_in_est, i_c_in_est, i_n_in_est]
feed_TD = [0.09, 0.09, 0.051642574, 0.27415651, 1.434984812, 0.109214796, 0.368394, 2.711362661, 0.0, 0.28707188, 0.21771537, 0.10664253, 0.004, 0.004495235714285714, 0.005, 0.0]
# [ch_i_in, pro_i_in, ch_in, pro_in, lip_in, gl_in, aa_in, lcfa_in, total_val_in, total_bu_in, total_pro_in, total_ac_in, i_c_in, i_n_in, X_lip_in, X_su_in, X_aa_in, X_lcfa_in, X_va_in, X_bu_in, X_pro_in, X_ac_in]
feed_TEA = [0.09, 0.09, 0.051642574, 0.27415651, 1.434984812, 0.109214796, 0.368394, 2.711362661, 0, 0.28707188, 0.21771537, 0.10664253, 0.004, 0.0044952357142857, 2E-3, 0, 2E-3, 0, 2E-3, 0, 2E-3, 0]


# Defining the results
# Anaerobic reactor
t_inicial = 0
t_final = 0.25/24
tiempo_an = []
tiempo_ae = []
v_liq_an = []
v_liq_ae = []
v_tot = []
lcfa = []
va_tot = []
bu_tot = []
pro_tot = []
ac_tot = []
metano = []
biogas = []
pH = []
nat = []
nat_tea = []
tie = []
#Aerobic reactor
alkalinity = []
oxygen = []
ammonia = []
nitrite = []
nitrate = []
cod_total = []
cod_soluble = []
dbp = []

def continuar(parameters, constants, inlet, t_inicial, t_final, initial):

    # Instantiation of Tanque_Substrate class
    tanquesustrato = st.Tanque_Sustrato(constants[0], initial[0], [t_inicial, t_final], np.linspace(t_inicial, t_final, int((t_final-t_inicial)/0.001)))
    ts_resultado = tanquesustrato.ode()
    inlet_td = [ts_resultado[2][-1], ts_resultado[3][-1], ts_resultado[4][-1], ts_resultado[5][-1], ts_resultado[6][-1], ts_resultado[9][-1], ts_resultado[10][-1], ts_resultado[11][-1], ts_resultado[12][-1], ts_resultado[13][-1], ts_resultado[14][-1], ts_resultado[15][-1], ts_resultado[7][-1], ts_resultado[8][-1], ts_resultado[22][-1], ts_resultado[23][-1]]
    y0_TS = [ts_resultado[1][-1],ts_resultado[2][-1],ts_resultado[3][-1],ts_resultado[4][-1],ts_resultado[5][-1],ts_resultado[6][-1],ts_resultado[7][-1],ts_resultado[8][-1],ts_resultado[9][-1],ts_resultado[10][-1],ts_resultado[11][-1],ts_resultado[12][-1],ts_resultado[13][-1],ts_resultado[14][-1],ts_resultado[15][-1],ts_resultado[22][-1],ts_resultado[23][-1]]
    ctes_TD[22] = ctes_TS[1]
    ctes_TA[0] = ctes_TS[0]*0
    # Instantiation of Tanque_Digestor class
    tanquedigestor = an.Tanque_Digestor(parameters[0], constants[1], inlet[0], initial[1], [t_inicial, t_final], np.linspace(t_inicial, t_final, int((t_final-t_inicial)/0.001)))
    td_resultado = tanquedigestor.ode()
    inlet_tea = [td_resultado[2][-1], td_resultado[3][-1], td_resultado[4][-1], td_resultado[5][-1], td_resultado[6][-1], td_resultado[7][-1], td_resultado[8][-1], td_resultado[9][-1], td_resultado[10][-1], td_resultado[11][-1], td_resultado[12][-1], td_resultado[13][-1], td_resultado[16][-1], td_resultado[18][-1], td_resultado[19][-1], td_resultado[20][-1], td_resultado[21][-1], td_resultado[22][-1], td_resultado[23][-1], td_resultado[24][-1], td_resultado[25][-1], td_resultado[26][-1]]
    y0_TD = [td_resultado[1][-1],td_resultado[2][-1],td_resultado[3][-1],td_resultado[4][-1],td_resultado[5][-1],td_resultado[6][-1],td_resultado[7][-1],td_resultado[8][-1],td_resultado[9][-1],td_resultado[10][-1],
         td_resultado[11][-1],td_resultado[12][-1],td_resultado[13][-1],td_resultado[14][-1],td_resultado[15][-1],td_resultado[16][-1],td_resultado[17][-1],td_resultado[18][-1],td_resultado[19][-1],td_resultado[20][-1],
         td_resultado[21][-1],td_resultado[22][-1],td_resultado[23][-1],td_resultado[24][-1],td_resultado[25][-1],td_resultado[26][-1],td_resultado[27][-1],td_resultado[28][-1],td_resultado[29][-1],td_resultado[30][-1],
         td_resultado[31][-1],td_resultado[32][-1],td_resultado[33][-1],td_resultado[34][-1],td_resultado[35][-1],td_resultado[36][-1],td_resultado[37][-1],td_resultado[38][-1],td_resultado[40][-1],td_resultado[41][-1],
         td_resultado[46][-1], td_resultado[47][-1], td_resultado[48][-1], td_resultado[49][-1]]
    ctes_TEA[0] = ctes_TD[23]
    # Instantiation of Tanque_EfluenteAnaerobio class
    tanqueefluente = anef.Tanque_EfluenteAnaerobio(constants[2], inlet[1], initial[2], [t_inicial, t_final], np.linspace(t_inicial, t_final, int((t_final-t_inicial)/0.001)))
    tanef_resultado = tanqueefluente.ode()
    inlet_ta = [tanef_resultado[28][-1], tanef_resultado[27][-1], tanef_resultado[26][-1], tanef_resultado[24][-1], ts_resultado[7][-1], ts_resultado[21][-1], ts_resultado[18][-1], ts_resultado[16][-1], 0.004, 1E-12, 1E-12]
    y0_TEA = [tanef_resultado[1][-1],tanef_resultado[2][-1],tanef_resultado[3][-1],tanef_resultado[4][-1],tanef_resultado[5][-1],tanef_resultado[6][-1],tanef_resultado[7][-1],tanef_resultado[8][-1],tanef_resultado[9][-1],tanef_resultado[10][-1],
        tanef_resultado[11][-1],tanef_resultado[12][-1],tanef_resultado[13][-1],tanef_resultado[14][-1],tanef_resultado[15][-1],tanef_resultado[16][-1],tanef_resultado[17][-1],tanef_resultado[18][-1],tanef_resultado[19][-1],tanef_resultado[20][-1],
        tanef_resultado[21][-1],tanef_resultado[22][-1],tanef_resultado[23][-1]]
    ctes_TA[1] = ctes_TEA[1]*1
    # Instantiation of Tanque_Aerobio class
    tanqueaerobio = asm.Tanque_Aerobio(parameters[1], constants[3], inlet[2], initial[3], [t_inicial, t_final], np.linspace(t_inicial, t_final, int((t_final-t_inicial)/0.001)))
    ta_resultado = tanqueaerobio.ode()
    y0_TA = [ta_resultado[1][-1],ta_resultado[2][-1],ta_resultado[3][-1],ta_resultado[4][-1],ta_resultado[5][-1],ta_resultado[6][-1],ta_resultado[7][-1],ta_resultado[8][-1],ta_resultado[9][-1],ta_resultado[10][-1],
         ta_resultado[11][-1],ta_resultado[12][-1],ta_resultado[13][-1],ta_resultado[14][-1],ta_resultado[15][-1],ta_resultado[21][-1],ta_resultado[22][-1],ta_resultado[23][-1], ta_resultado[24][-1]]
    
    
    # Saving the results
    tiempo_an.append(td_resultado[0])
    tiempo_ae.append(ta_resultado[0])
    v_liq_an.append(td_resultado[1])
    v_liq_ae.append(ta_resultado[1])
    v_tot.append(td_resultado[43])
    lcfa.append(td_resultado[9])
    va_tot.append(td_resultado[10])
    bu_tot.append(td_resultado[11])
    pro_tot.append(td_resultado[12])
    ac_tot.append(td_resultado[13])
    biogas.append(td_resultado[41])
    metano.append(td_resultado[40])
    pH.append(td_resultado[44])
    nat.append(td_resultado[18]*14000)
    tie.append(td_resultado[45])
    alkalinity.append(ta_resultado[3])
    oxygen.append(ta_resultado[4])
    ammonia.append(ta_resultado[5]*1000)
    nitrite.append(ta_resultado[6])
    nitrate.append(ta_resultado[7]*1000)
    cod_total.append(ta_resultado[16])
    cod_soluble.append(ta_resultado[17]*1000)
    nat_tea.append(tanef_resultado[27]*1000)
    dbp.append(ta_resultado[15])
    
    # Saving new initial conditions
    return [y0_TS, y0_TD, y0_TEA, y0_TA, inlet_td, inlet_tea, inlet_ta]

while t_final < 4:
    
    y0_TS, y0_TD, y0_TEA, y0_TA, feed_TD, feed_TEA, feed_TA = continuar([pars_TD, pars_TA], [ctes_TS, ctes_TD, ctes_TEA, ctes_TA], [feed_TD, feed_TEA, feed_TA], t_inicial, t_final, [y0_TS, y0_TD, y0_TEA, y0_TA]) #Llenado    
    t_inicial = t_final
    t_final = t_inicial + 2.5/24 #1.5
    ctes_TS[1] = 0
    ctes_TD[36] = 1
    ctes_TS[0] = 0
    ctes_TA[5] = 1
    ctes_TEA[1] = 0
    pars_TA[7] = 1000
    y0_TEA[0] = 0.01
    y0_TS, y0_TD, y0_TEA, y0_TA, feed_TD, feed_TEA, feed_TA = continuar([pars_TD, pars_TA], [ctes_TS, ctes_TD, ctes_TEA, ctes_TA], [feed_TD, feed_TEA, feed_TA], t_inicial, t_final, [y0_TS, y0_TD, y0_TEA, y0_TA]) #Reacción Aerobia
    t_inicial = t_final
    t_final = t_inicial + 2.5/24 #1.2
    pars_TA[7] = 0
    ctes_TA[9] = 0.035
    y0_TS, y0_TD, y0_TEA, y0_TA, feed_TD, feed_TEA, feed_TA = continuar([pars_TD, pars_TA], [ctes_TS, ctes_TD, ctes_TEA, ctes_TA], [feed_TD, feed_TEA, feed_TA], t_inicial, t_final, [y0_TS, y0_TD, y0_TEA, y0_TA]) #Reacción Anóxica
    t_inicial = t_final
    t_final = t_inicial + 0.5/24
    #pars_TA[7] = 0
    ctes_TA[5] = 0
    ctes_TA[9] = 2.75
    y0_TS, y0_TD, y0_TEA, y0_TA, feed_TD, feed_TEA, feed_TA = continuar([pars_TD, pars_TA], [ctes_TS, ctes_TD, ctes_TEA, ctes_TA], [feed_TD, feed_TEA, feed_TA], t_inicial, t_final, [y0_TS, y0_TD, y0_TEA, y0_TA]) #Decantación
    t_inicial = t_final
    t_final = t_inicial + 0.25/24
    ctes_TA[2] = 194
    y0_TS, y0_TD, y0_TEA, y0_TA, feed_TD, feed_TEA, feed_TA = continuar([pars_TD, pars_TA], [ctes_TS, ctes_TD, ctes_TEA, ctes_TA], [feed_TD, feed_TEA, feed_TA], t_inicial, t_final, [y0_TS, y0_TD, y0_TEA, y0_TA]) #Vaciado
    t_inicial = t_final
    t_final = t_inicial + 17.25/24
    ctes_TA[2] = 0
    y0_TS, y0_TD, y0_TEA, y0_TA, feed_TD, feed_TEA, feed_TA = continuar([pars_TD, pars_TA], [ctes_TS, ctes_TD, ctes_TEA, ctes_TA], [feed_TD, feed_TEA, feed_TA], t_inicial, t_final, [y0_TS, y0_TD, y0_TEA, y0_TA]) #Reacción Anaerobia
    t_inicial = t_final
    t_final = t_inicial + 0.5/24
    ctes_TD[36] = 0
    y0_TS, y0_TD, y0_TEA, y0_TA, feed_TD, feed_TEA, feed_TA = continuar([pars_TD, pars_TA], [ctes_TS, ctes_TD, ctes_TEA, ctes_TA], [feed_TD, feed_TEA, feed_TA], t_inicial, t_final, [y0_TS, y0_TD, y0_TEA, y0_TA]) #Decantación
    t_inicial = t_final
    t_final = t_inicial + 0.25/24
    ctes_TD[23] = 264
    y0_TS, y0_TD, y0_TEA, y0_TA, feed_TD, feed_TEA, feed_TA = continuar([pars_TD, pars_TA], [ctes_TS, ctes_TD, ctes_TEA, ctes_TA], [feed_TD, feed_TEA, feed_TA], t_inicial, t_final, [y0_TS, y0_TD, y0_TEA, y0_TA]) #Vaciado
    t_inicial = t_final
    t_final = t_inicial + 0.25/24
    ctes_TD[23] = 0
    ctes_TS[1] = 264
    ctes_TEA[1] = 194
    y0_TD[14] = 1E-12
    y0_TD[16] = 1E-6
    y0_TD[38] = 1E-12
    y0_TD[39] = 1E-12  
    ctes_TS[0] = 194 # CICLO 1
    
    

data_nat0 = pd.read_excel(r'Datos.xlsx', sheet_name='NAT0')
nat0 = pd.DataFrame(data_nat0, columns=['NAT [mg/L]'])
cods0 = pd.DataFrame(data_nat0, columns=['CODS [mg/L]'])
nitrate0 = pd.DataFrame(data_nat0, columns=['Nitrato [mg/L]'])
tiempo_nat0 = pd.DataFrame(data_nat0, columns=['Tiempo [d]'])
#tiempo_nat0 = round(tiempo_nat0, 4)

tiempo_an = [i for sublist in tiempo_an for i in sublist]
tiempo_ae = [i for sublist in tiempo_ae for i in sublist]

print([i for sublist in ammonia for i in sublist][-1])

"""
while intervalo < 7:
    # Saving new initial condition
    y0 = continuar(pars_TD, ctes_TD, feed_TD, intervalo, y0)
    # Changing intervalo
    intervalo += 1
    # Restarting conditions (feeding, temperature, etc)
    print("Check")
"""
    
"""
np.savetxt('data.txt',data,delimiter=',')
"""

# Plot the inputs and results
# Gráfica 16
plt.figure(figsize=(8,5.5))
plt.vlines(x=[3+0.25/24], ymin=0, ymax=3000, colors=['black'], linestyles=['-'])
plt.vlines(x=[3+0.25/24+2.5/24], ymin=0, ymax=3000, colors=['black'], linestyles=['-'])
plt.vlines(x=[3+0.25/24+2.5/24+2.5/24], ymin=0, ymax=3000, colors=['black'], linestyles=['-'])
plt.text(x=3+0.5/24, y=2750, s='Reacción aerobia', fontsize=10, color='black')
plt.text(x=3+0.25/24+3/24, y=2750, s='Reacción anóxica', fontsize=10, color='black')
plt.plot(tiempo_nat0, cods0, 'o', color='green', ms=5, label='Datos experimentales')
plt.plot(tiempo_ae, [i for sublist in cod_soluble for i in sublist], '-', color='brown', lw=2.5, label='MAn2')
plt.xlabel('Tiempo, d', fontsize=15)
plt.ylim(0, 3000)
plt.ylabel('DQO'+'\u209B'+', mg DQO L' + r'$^{-1}$', fontsize=15)
#plt.legend(loc='lower left', fontsize=10)
plt.show()

# Gráfica 17
plt.figure(figsize=(8,5.5))
plt.vlines(x=[3+0.25/24], ymin=0, ymax=200, colors=['black'], linestyles=['-'])
plt.vlines(x=[3+0.25/24+2.5/24], ymin=0, ymax=200, colors=['black'], linestyles=['-'])
plt.vlines(x=[3+0.25/24+2.5/24+2.5/24], ymin=0, ymax=200, colors=['black'], linestyles=['-'])
plt.text(x=3+0.5/24, y=190, s='Reacción aerobia', fontsize=10, color='black')
plt.text(x=3+0.25/24+3/24, y=190, s='Reacción anóxica', fontsize=10, color='black')
plt.plot(tiempo_nat0, nat0, 'o', color='green', ms=5, label='Datos experimentales')
plt.plot(tiempo_ae, [i for sublist in ammonia for i in sublist], '-', color='brown', lw=2.5, label='MAn2')
plt.xlabel('Tiempo, d', fontsize=15)
plt.ylim(0, 200)
plt.ylabel('NAT, mg N-NH'+'\u2083'+' L' + r'$^{-1}$', fontsize=15)
#plt.legend(loc='lower left', fontsize=10)
plt.show()

# Gráfica 18
plt.figure(figsize=(8,5.5))
plt.vlines(x=[3+0.25/24], ymin=0, ymax=20, colors=['black'], linestyles=['-'])
plt.vlines(x=[3+0.25/24+2.5/24], ymin=0, ymax=20, colors=['black'], linestyles=['-'])
plt.vlines(x=[3+0.25/24+2.5/24+2.5/24], ymin=0, ymax=20, colors=['black'], linestyles=['-'])
plt.text(x=3+0.5/24, y=15, s='Reacción aerobia', fontsize=10, color='black')
plt.text(x=3+0.25/24+3/24, y=15, s='Reacción anóxica', fontsize=10, color='black')
plt.plot(tiempo_nat0, nitrate0, 'o', color='green', ms=5, label='Datos experimentales')
plt.plot(tiempo_ae, [i for sublist in nitrate for i in sublist], '-', color='brown', lw=2.5, label='MAn2')
plt.xlabel('Tiempo, d', fontsize=15)
plt.ylim(-0.3, 60)
plt.ylabel('Nitratos, mg N-NO'+'\u2083'+' L' + r'$^{-1}$', fontsize=15)
#plt.legend(loc='lower left', fontsize=10)
plt.show()

plt.figure(figsize=(8,5.5))
#plt.plot(tiempo_datos, nitrate_datos, 'o', color='green', ms=5, label='Datos experimentales')
plt.plot(tiempo_ae, [i for sublist in dbp for i in sublist], '-', color='orange', lw=2.5, label='MAn2')
plt.xlabel('Tiempo, d', fontsize=15)
#plt.ylim(0.2, 0.4)
plt.ylabel('Nitratos, mg N-NO'+'\u2083'+' L' + r'$^{-1}$', fontsize=15)
#plt.legend(loc='lower left', fontsize=10)
plt.show()