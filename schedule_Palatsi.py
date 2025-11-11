import Palatsi as pa
import Substrate_tank as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# Initial Conditions for the States [26]
#[V_liq0, ch_i0, pro_i0, ch0, pro0, lip0, gl0, aa0, lcfa0, va_tot0, bu_tot0, pro_tot0, ac_tot0, ch40, ch4_g0, i_c0, co2_gas0, i_n0, x_lip0, x_gl0, x_aa0, x_lcfa0, x_va0, x_bu0, x_pro0, x_ac0, Ta]
y0_TD = [2.25, 0, 0, 0, 0, 0, 0, 0, 0, 0.00162, 0.00019, 0.0003, 0.00011, 0.00069, 1e-12, 1e-12, 1e-12, 1e-12, 0.5/100, 1e-06, 0.05/14, 1E-12, 1E-12, 1E-12, 1E-12, 1E-12, 1E-12, 1E-12, 0.5, (55/6)*0.5, (55/6)*0.5, (55/9)*0.5, (55/9)*0.5, (55/9)*0.5, (55/3)*0.5, (55/3)*0.5, 0.0371, 1E-3, 0.0, 1E-12, 1E-12]
#[V_liq0, ch_i0, pro_i0, ch0, pro0, lip0, i_c0, i_n0, gl0, aa0, lcfa0, va_tot0, bu_tot0, pro_tot0, ac_tot0, cat0, an0]
y0_TS = [10000, 0.09, 0.09, 0.051642574, 0.27415651, 1.434984812, 0.4/100, 0.0629333/14, 0.109214796, 0.368394, 2.711362661, 0, 0.28707188, 0.21771537, 0.10664253, 1E-2, 0]



# Constants
# [f_i_ch, f_i_pro, f_lip_lcfa, f_lip_pro, f_bu_gl, f_pro_gl, f_ac_gl, f_va_aa, f_bu_aa, f_pro_aa, f_ac_aa, f_ac_lcfa, f_ch4_lcfa, f_pro_va, f_ac_va, f_ch4_va, f_ac_bu, f_ch4_bu, f_ac_pro, f_ch4_pro, f_ch4_ac, f_bio_ch, f_bio_prot, qin, qout, n_bio, n_aa, c_lip, c_gl, c_aa, c_lcfa, c_va, c_bu, c_pro_ c_ac, V_gas, qin_est]
ctes_TD = [0.25, 0.1, 0.2, 0.2, 0.25, 0.95, 0.13, 0.27, 0.41, 0.23, 0.26, 0.05, 0.4, 0.7, 0.54, 0.31, 0.8, 0.57, 0.19, 0.06, 0.3, 0.15, 0.2, 0.43, 264, 0, 0.00625, 0.007, 0, 0.0313, 0.032, 0.0313, 0.0240, 0.0250, 0.0268, 0.0313, 0.0156, 0.01, 0, 0, 1026, 30, 87.5, 0.0015]
# [qby, qout]
ctes_TS = [0, 264]


# Kinetics parameters
# [k_hyd_pro, k_hyd_ch, k_dead, ks_in, ki_vfa, ki_lcfa, ki_ac_va, ki_ac_bu, ki_ac_pro, ki_nh3, pH_ul, pH_ll, umax_acid_lip, ks_lip, y_lip, umax_acid_su, ks_gl, y_gl, umax_acid_aa, ks_aa, y_aa, umax_acid_lcfa, ks_lcfa, y_lcfa, umax_acid_val, ks_va, y_va, umax_acid_but, ks_bu, y_bu, umax_acid_pro, ks_pro, y_pro, umax_met, ks_ac, y_ac, deltaH_CH4, deltaH_CO2, shg, tau, temp, kH298_CH4, kH298_CO2, kLa]
pars_TD = [0.4, 0.2, 0.2, 0.25, 0.02, 1e-4, 5E-6, 1E-5, 3.5E-6, 2.73E3, 0.0018, 5.5, 4, 6, 5, 7, 6, 3, 0.5, 0.1, 4, 0.3, 0.08, 0.36, 0.4, 0.06, 0.12, 0.3, 0.06, 0.52, 0.3, 0.04, 2.1, 2E-5, 0.06, 0.4, 0.15, 0.05, -14240, -19410, -4180, 308.15, 0.00142, 0.03545, 0.00079, 200] #44


# Feed concentrations
# [ch_i_in_hidro pro_i_in_hidro, ch_in_hidro, pro_in_hidro, lip_in_hidro, gl_in, aa_in, lcfa_in, total_val_in, total_bu_in, total_pro_in, total_ac_in, i_c_in_hidro, i_n_in_hidro, X_lip_in, X_su_in, X_aa_in, X_lcfa_in, ch_i_in_est, pro_i_in_est, ch_in_est, pro_in_est, lip_in_est, i_c_in_est, i_n_in_est]
feed_TD = [0.09, 0.09, 0.051642574, 0.27415651, 1.434984812, 0.109214796, 0.368394, 2.711362661, 0, 0.28707188, 0.21771537, 0.10664253, 0.004, 0.0044952357142857, 2E-3, 0]


# Defining the results
t_inicial = 0
t_final = 0.25/24
tiempo = []
v_liq = []
v_tot = []
carbo = []
prote = []
lipi = []
sugars = []
aminoacids = []
lcfa = []
va_tot = []
bu_tot = []
pro_tot = []
ac_tot = []
tac = []
fos = []
metano = []
biogas = []
pH = []
sv = []
nat = []
pH_TH = []
sv_TH = []
tie = []

def continuar(parameters, constants, inlet, t_inicial, t_final, initial):

    # Instantiation of Tanque_Substrate class
    tanquesustrato = st.Tanque_Sustrato(constants[0], initial[0], [t_inicial, t_final], np.linspace(t_inicial, t_final, int((t_final-t_inicial)/0.001)))
    ts_resultado = tanquesustrato.ode()
    inlet_td = [ts_resultado[2][-1], ts_resultado[3][-1], ts_resultado[4][-1], ts_resultado[5][-1], ts_resultado[6][-1], ts_resultado[9][-1], ts_resultado[10][-1], ts_resultado[11][-1], ts_resultado[12][-1], ts_resultado[13][-1], ts_resultado[14][-1], ts_resultado[15][-1], ts_resultado[7][-1], ts_resultado[8][-1], ts_resultado[22][-1], ts_resultado[23][-1]]
    y0_TS = [ts_resultado[1][-1],ts_resultado[2][-1],ts_resultado[3][-1],ts_resultado[4][-1],ts_resultado[5][-1],ts_resultado[6][-1],ts_resultado[7][-1],ts_resultado[8][-1],ts_resultado[9][-1],ts_resultado[10][-1],ts_resultado[11][-1],ts_resultado[12][-1],ts_resultado[13][-1],ts_resultado[14][-1],ts_resultado[15][-1],ts_resultado[22][-1],ts_resultado[23][-1]]
    ctes_TD[22] = ctes_TS[1]
    # Instantiation of Tanque_Digestor class
    tanquedigestor = pa.Tanque_Digestor(parameters, constants[1], inlet, initial[1], [t_inicial, t_final], np.linspace(t_inicial, t_final, int((t_final-t_inicial)/0.001)))
    td_resultado = tanquedigestor.ode()
    
    y0_TD = [td_resultado[1][-1],td_resultado[2][-1],td_resultado[3][-1],td_resultado[4][-1],td_resultado[5][-1],td_resultado[6][-1],td_resultado[7][-1],td_resultado[8][-1],td_resultado[9][-1],td_resultado[10][-1],
         td_resultado[11][-1],td_resultado[12][-1],td_resultado[13][-1],td_resultado[14][-1],td_resultado[15][-1],td_resultado[16][-1],td_resultado[17][-1],td_resultado[18][-1],td_resultado[19][-1],td_resultado[20][-1],
         td_resultado[21][-1],td_resultado[22][-1],td_resultado[23][-1],td_resultado[24][-1],td_resultado[25][-1],td_resultado[26][-1],td_resultado[27][-1],td_resultado[28][-1],td_resultado[29][-1],td_resultado[30][-1],
         td_resultado[31][-1],td_resultado[32][-1],td_resultado[33][-1],td_resultado[34][-1],td_resultado[35][-1],td_resultado[36][-1],td_resultado[37][-1],td_resultado[38][-1],td_resultado[39][-1],td_resultado[41][-1],td_resultado[42][-1]]
    # Saving the results
    tiempo.append(td_resultado[0])
    v_liq.append(td_resultado[1])
    v_tot.append(td_resultado[51])
    carbo.append(td_resultado[4])
    prote.append(td_resultado[5])
    lipi.append(td_resultado[6])
    sugars.append(td_resultado[7])
    aminoacids.append(td_resultado[7])
    lcfa.append(td_resultado[9])
    va_tot.append(td_resultado[10])
    bu_tot.append(td_resultado[11])
    pro_tot.append(td_resultado[12])
    ac_tot.append(td_resultado[13])
    biogas.append(td_resultado[41])
    metano.append(td_resultado[42])
    pH.append(td_resultado[53])
    nat.append(td_resultado[31])
    fos.append(td_resultado[32])
    tac.append(td_resultado[33])
    sv.append(td_resultado[34])
    tie.append(td_resultado[54])
    
    # Saving new initial conditions
    return [y0_TS, y0_TD, inlet_td]

while t_final < 10:
    y0_TS, y0_TD, feed_TD = continuar(pars_TD, [ctes_TS, ctes_TD], feed_TD, t_inicial, t_final, [y0_TS, y0_TD]) #Llenado
    t_inicial = t_final
    t_final = t_inicial + 23/24
    ctes_TS[1] = 0
    ctes_TD[39] = 1
    y0_TS, y0_TD, feed_TD = continuar(pars_TD, [ctes_TS, ctes_TD], feed_TD, t_inicial, t_final, [y0_TS, y0_TD]) #Reacción
    t_inicial = t_final
    t_final = t_inicial + 0.5/24
    ctes_TD[39] = 0
    y0_TS, y0_TD, feed_TD = continuar(pars_TD, [ctes_TS, ctes_TD], feed_TD, t_inicial, t_final, [y0_TS, y0_TD]) #Decantación
    t_inicial = t_final
    t_final = t_inicial + 0.25/24
    ctes_TD[25] = 264
    y0_TS, y0_TD, feed_TD = continuar(pars_TD, [ctes_TS, ctes_TD], feed_TD, t_inicial, t_final, [y0_TS, y0_TD]) #Vaciado
    t_inicial = t_final
    t_final = t_inicial + 0.25/24
    ctes_TD[25] = 0
    ctes_TS[1] = 264
    y0_TD[15] = 1E-12
    y0_TD[17] = 1E-12
    y0_TD[19] = 1E-6
    y0_TD[39] = 1E-12
    y0_TD[40] = 1E-12
 

data_biogas = pd.read_excel(r'Datos.xlsx', sheet_name='Biogas')
data_metano = pd.read_excel(r'Datos.xlsx', sheet_name='Metano')
biogas_angelidaki = pd.DataFrame(data_biogas, columns=['Biogas [L]'])
metano_angelidaki = pd.DataFrame(data_metano, columns=['CH4 [L]'])
tiempo_angelidaki = pd.DataFrame(data_biogas, columns=['Tiempo [d]'])
tiempo_angelidaki = round(tiempo_angelidaki, 3)

tiempo = [round(i,3) for sublist in tiempo for i in sublist]


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
plt.figure()
plt.plot(tiempo, [i for sublist in metano for i in sublist], 'g-', color= "green", ms=5)
plt.xlabel('Tiempo, d', fontsize=15)
plt.ylim(0, 10)
plt.ylabel('AGCL, mg/L', fontsize=15)
"""
plt.subplot(2,2,1)
plt.plot(tiempo_angelidaki, metano_angelidaki, 'o', color= "green", ms=5)
plt.ylabel('pH')
plt.legend(['pH'],loc='best')

plt.subplot(2,2,2)
plt.plot([i for sublist in tie for i in sublist] ,[i for sublist in pH for i in sublist],'g-',linewidth=3)
plt.ylabel('L')
plt.legend(['Methane'],loc='best')

plt.subplot(2,2,3)
plt.plot(tiempo, [i for sublist in biogas for i in sublist],'y-',linewidth=3)
plt.plot(tiempo, [i for sublist in metano for i in sublist], 'g-', color= "green", ms=5)
plt.ylabel('gCOD/L')
plt.legend(['Ác. Propiónico'],loc='best')

plt.subplot(2,2,4)
plt.plot(tiempo_angelidaki, metano_angelidaki, 'o', color= "green", ms=5)
plt.plot(tiempo, [i for sublist in metano for i in sublist],'r-',linewidth=3)
"""

plt.show()

