import ASM as asm
import Substrate_tank as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])



# Initial Conditions for the States [26]
#[V_liq0, Ss0, alk0, o20, nh40, no20, no30, x_S_film0, x_I_film0, x_H_film0, x_A_film0, x_N_film0, x_sto_film0, x_ana_film0, V_sol0, dbp0]
y0_TA = [1.7500000000011884, 0.17661701827958654, 0.020441154656642587, -1.9734021732376204e-21, 0.039356752543551483, 4.712153898978602e-15, -2.864890261424499e-17, 51.14698947310692, 1.6626823899583525, 23.03027206373868, 0.7943243797166637, 0.6442968971277878, 7.342309356291359, 0.7944502866488288, 0.025, 1e-12, 1e-12, 1e-12, 0.2]
#[V_liq0, ch_i0, pro_i0, ch0, pro0, lip0, i_c0, i_n0, gl0, aa0, lcfa0, va_tot0, bu_tot0, pro_tot0, ac_tot0, cat0, an0]
y0_TS = [10000, 0.09, 0.09, 0.051642574, 0.27415651, 1.434984812, 0.4/100, 0.0629333/14, 0.109214796, 0.368394, 2.711362661, 0, 0.28707188, 0.21771537, 0.10664253, 1E-2, 0]


# Constants
# [qby, qan, qout, V_gas, qs, lamb, rho_s, AC_bp, MC_bp, k_det, o2s, f_X, f_N_I, f_N_B, f_S_I, f_N_Ss]
ctes_TA = [0, 0, 0, 0.01, 0, 0, 1026, 30, 87.5, 2.75, 0.009, 0.2, 0, 0.08, 0, 0.09]
# [qby, qout]
ctes_TS = [194, 0]


# Kinetics parameters
# [b_A, b_N, b_H, b_sto, ks_X, n_g, k_alk, kLa, ks_sto, ks_ss, k_nh4, kA_no2, kA_no3, kN_no2, kN_no3, kh_no2, kh_no3, k_A_o2, k_N_o2, k_H_o2, umax_A, umax_N, umax_H, k_sto, y_A, y_N, y_H, y_hsto, y_sto, k_h_s, ki_no2]
pars_TA = [0.18, 0.16, 0.26, 0.26, 2.45, 0.6, 0.006, 0, 1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0005, 0.0005, 0.00074, 0.00175, 0.0002, 1.10, 0.96, 3.76, 2.3, 0.18, 0.08, 0.57, 0.68, 0.8, 3, 0.25] #31


# Feed concentrations
# [alk_an, nh4_an, x_S_an, Ss_an, alk_by, nh4_by, x_S_by, Ss_by, o2_in, no2_in, no3_in]
feed_TA = [0, 0, 0, 0, 0.004, 0.0629333, 1.9407838960000001, 3.800401237, 0.004, 1e-12, 1e-12]


# Defining the results
t_inicial = 0
t_final = 0.25/24
tiempo = []
v_liq = []
v_tot = []
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
    inlet_ta = [0, 0, 0, 0, ts_resultado[7][-1], ts_resultado[21][-1], ts_resultado[18][-1], ts_resultado[16][-1], 0.004, 1E-12, 1E-12]
    y0_TS = [ts_resultado[1][-1],ts_resultado[2][-1],ts_resultado[3][-1],ts_resultado[4][-1],ts_resultado[5][-1],ts_resultado[6][-1],ts_resultado[7][-1],ts_resultado[8][-1],ts_resultado[9][-1],ts_resultado[10][-1],ts_resultado[11][-1],ts_resultado[12][-1],ts_resultado[13][-1],ts_resultado[14][-1],ts_resultado[15][-1],ts_resultado[22][-1],ts_resultado[23][-1]]
    ctes_TA[0] = ctes_TS[0]
    # Instantiation of Tanque_Digestor class
    tanqueaerobio = asm.Tanque_Aerobio(parameters, constants[1], inlet, initial[1], [t_inicial, t_final], np.linspace(t_inicial, t_final, int((t_final-t_inicial)/0.001)))
    ta_resultado = tanqueaerobio.ode()
    
    y0_TA = [ta_resultado[1][-1],ta_resultado[2][-1],ta_resultado[3][-1],ta_resultado[4][-1],ta_resultado[5][-1],ta_resultado[6][-1],ta_resultado[7][-1],ta_resultado[8][-1],ta_resultado[9][-1],ta_resultado[10][-1],
         ta_resultado[11][-1],ta_resultado[12][-1],ta_resultado[13][-1],ta_resultado[14][-1],ta_resultado[15][-1], ta_resultado[21][-1],
         ta_resultado[22][-1],ta_resultado[23][-1],ta_resultado[24][-1]]
    # Saving the results
    tiempo.append(ta_resultado[0])
    v_liq.append(ta_resultado[1])
    alkalinity.append(ta_resultado[3])
    oxygen.append(ta_resultado[4])
    ammonia.append(ta_resultado[5]*1000)
    nitrite.append(ta_resultado[6])
    nitrate.append(ta_resultado[7])
    cod_total.append(ta_resultado[16])
    cod_soluble.append(ta_resultado[17]*1000)
    dbp.append(ta_resultado[15])
    
    
    # Saving new initial conditions
    return [y0_TS, y0_TA, inlet_ta]

while t_final < 0.7:
    y0_TS, y0_TA, feed_TA = continuar(pars_TA, [ctes_TS, ctes_TA], feed_TA, t_inicial, t_final, [y0_TS, y0_TA]) #Llenado
    t_inicial = t_final
    t_final = t_inicial + 2.5/24
    ctes_TS[0] = 0
    ctes_TA[5] = 1
    pars_TA[7] = 1000
    ctes_TA[9] = 2.75
    y0_TS, y0_TA, feed_TA = continuar(pars_TA, [ctes_TS, ctes_TA], feed_TA, t_inicial, t_final, [y0_TS, y0_TA]) #Reacción
    t_inicial = t_final
    t_final = t_inicial + 2.5/24
    pars_TA[7] = 0
    ctes_TA[9] = 0.035
    y0_TS, y0_TA, feed_TA = continuar(pars_TA, [ctes_TS, ctes_TA], feed_TA, t_inicial, t_final, [y0_TS, y0_TA]) #Reacción
    t_inicial = t_final
    t_final = t_inicial + 0.5/24
    ctes_TA[5] = 0
    y0_TS, y0_TA, feed_TA = continuar(pars_TA, [ctes_TS, ctes_TA], feed_TA, t_inicial, t_final, [y0_TS, y0_TA]) #Decantación
    t_inicial = t_final
    t_final = t_inicial + 0.25/24
    ctes_TA[2] = 194
    y0_TS, y0_TA, feed_TA = continuar(pars_TA, [ctes_TS, ctes_TA], feed_TA, t_inicial, t_final, [y0_TS, y0_TA]) #Vaciado
    ctes_TA[2] = 0
    """
    t_inicial = t_final
    t_final = t_inicial + 18/24
    ctes_TA[2] = 0
    y0_TS, y0_TA, feed_TA = continuar(pars_TA, [ctes_TS, ctes_TA], feed_TA, t_inicial, t_final, [y0_TS, y0_TA]) #Vaciado
    """
    t_inicial = t_final
    t_final = t_inicial + 0.25/24
    ctes_TS[0] = 194
    #print(tiempo[-1])



data = pd.read_excel(r'Datos.xlsx', sheet_name='NAT')
nat_datos = pd.DataFrame(data, columns=['NAT [mg/L]'])
nitrate_datos = pd.DataFrame(data, columns=['Nitrato [mg/L]'])
cods_datos = pd.DataFrame(data, columns=['CODS [mg/L]'])
tiempo_datos = pd.DataFrame(data, columns=['Tiempo [d]'])

print(y0_TA, feed_TA)

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
# Gráfica 7
plt.figure()
plt.vlines(x=[0.25/24], ymin=0, ymax=4000, colors=['black'], linestyles=['-'])
plt.vlines(x=[0.25/24+2.5/24], ymin=0, ymax=4000, colors=['black'], linestyles=['-'])
plt.vlines(x=[0.25/24+2.5/24+2.5/24], ymin=0, ymax=4000, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+0.25/24], ymin=0, ymax=4000, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+0.25/24+2.5/24], ymin=0, ymax=4000, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+0.25/24+2.5/24+2.5/24], ymin=0, ymax=4000, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+6/24+0.25/24], ymin=0, ymax=4000, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+6/24+0.25/24+2.5/24], ymin=0, ymax=4000, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+6/24+0.25/24+2.5/24+2.5/24], ymin=0, ymax=4000, colors=['black'], linestyles=['-'])
plt.text(x=0.4/24, y=3750, s='Reacción aerobia', fontsize=10, color='black')
plt.text(x=2.9/24, y=3750, s='Reacción anóxica', fontsize=10, color='black')
plt.text(x=6/24+0.4/24, y=3750, s='Reacción aerobia', fontsize=10, color='black')
plt.text(x=6/24+2.9/24, y=3750, s='Reacción anóxica', fontsize=10, color='black')
plt.text(x=6/24+6/24+0.4/24, y=3750, s='Reacción aerobia', fontsize=10, color='black')
plt.text(x=6/24+6/24+2.9/24, y=3750, s='Reacción anóxica', fontsize=10, color='black')
plt.plot(tiempo_datos, cods_datos, 'o', color='green', ms=5, label='Datos experimentales')
plt.plot([i for sublist in tiempo for i in sublist], [i for sublist in cod_soluble for i in sublist], '-', color='orange', lw=2.5, label='MAn2')
plt.xlabel('Tiempo, d', fontsize=15)
plt.ylim(0, 4000)
plt.ylabel('DQO'+'\u209B'+', mg L' + r'$^{-1}$', fontsize=15)
#plt.legend(loc='lower left', fontsize=10)
plt.show()

# Gráfica 8
plt.figure()
plt.vlines(x=[0.25/24], ymin=0, ymax=65, colors=['black'], linestyles=['-'])
plt.vlines(x=[0.25/24+2.5/24], ymin=0, ymax=65, colors=['black'], linestyles=['-'])
plt.vlines(x=[0.25/24+2.5/24+2.5/24], ymin=0, ymax=65, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+0.25/24], ymin=0, ymax=65, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+0.25/24+2.5/24], ymin=0, ymax=65, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+0.25/24+2.5/24+2.5/24], ymin=0, ymax=65, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+6/24+0.25/24], ymin=0, ymax=65, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+6/24+0.25/24+2.5/24], ymin=0, ymax=65, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+6/24+0.25/24+2.5/24+2.5/24], ymin=0, ymax=65, colors=['black'], linestyles=['-'])
plt.text(x=0.4/24, y=60, s='Reacción aerobia', fontsize=10, color='black')
plt.text(x=2.9/24, y=60, s='Reacción anóxica', fontsize=10, color='black')
plt.text(x=6/24+0.4/24, y=60, s='Reacción aerobia', fontsize=10, color='black')
plt.text(x=6/24+2.9/24, y=60, s='Reacción anóxica', fontsize=10, color='black')
plt.text(x=6/24+6/24+0.4/24, y=60, s='Reacción aerobia', fontsize=10, color='black')
plt.text(x=6/24+6/24+2.9/24, y=60, s='Reacción anóxica', fontsize=10, color='black')
plt.plot(tiempo_datos, nat_datos, 'o', color='green', ms=5, label='Datos experimentales')
plt.plot([i for sublist in tiempo for i in sublist], [i for sublist in ammonia for i in sublist], '-', color='orange', lw=2.5, label='MAn2')
plt.xlabel('Tiempo, d', fontsize=15)
plt.ylim(0, 65)
plt.ylabel('NAT, mg N-NH'+'\u2083'+' L' + r'$^{-1}$', fontsize=15)
#plt.legend(loc='lower left', fontsize=10)
plt.show()
"""

# Gráfica 9
plt.figure()
#plt.plot(tiempo_datos, nitrate_datos, 'o', color='green', ms=5, label='Datos experimentales')
plt.plot([i for sublist in tiempo for i in sublist], [i for sublist in dbp for i in sublist], '-', color='orange', lw=2.5, label='MAn2')
plt.xlabel('Tiempo, d', fontsize=15)
plt.ylim(0.2, 0.4)
plt.ylabel('Nitratos, mgN-NO'+'\u2083'+'/L', fontsize=15)
#plt.legend(loc='lower left', fontsize=10)
plt.show()
"""
plt.figure()
plt.vlines(x=[0.25/24], ymin=0, ymax=20, colors=['black'], linestyles=['-'])
plt.vlines(x=[0.25/24+2.5/24], ymin=0, ymax=20, colors=['black'], linestyles=['-'])
plt.vlines(x=[0.25/24+2.5/24+2.5/24], ymin=0, ymax=20, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+0.25/24], ymin=0, ymax=20, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+0.25/24+2.5/24], ymin=0, ymax=20, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+0.25/24+2.5/24+2.5/24], ymin=0, ymax=20, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+6/24+0.25/24], ymin=0, ymax=20, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+6/24+0.25/24+2.5/24], ymin=0, ymax=20, colors=['black'], linestyles=['-'])
plt.vlines(x=[6/24+6/24+0.25/24+2.5/24+2.5/24], ymin=0, ymax=20, colors=['black'], linestyles=['-'])
plt.text(x=0.4/24, y=17, s='Reacción aerobia', fontsize=10, color='black')
plt.text(x=2.9/24, y=17, s='Reacción anóxica', fontsize=10, color='black')
plt.text(x=6/24+0.4/24, y=17, s='Reacción aerobia', fontsize=10, color='black')
plt.text(x=6/24+2.9/24, y=17, s='Reacción anóxica', fontsize=10, color='black')
plt.text(x=6/24+6/24+0.4/24, y=17, s='Reacción aerobia', fontsize=10, color='black')
plt.text(x=6/24+6/24+2.9/24, y=17, s='Reacción anóxica', fontsize=10, color='black')
plt.plot(tiempo_datos, nitrate_datos, 'o', color='green', ms=5, label='Datos experimentales')
plt.plot([i for sublist in tiempo for i in sublist], [i for sublist in nitrate for i in sublist], '-', color='orange', lw=2.5, label='MAn2')
#plt.hlines(y=[10], xmin=min(tiempo_ae), xmax=max(tiempo_ae), colors=['black'], linestyles=['--'])
plt.xlabel('Tiempo, d', fontsize=15)
plt.ylim(-0.3, 20)
plt.ylabel('Nitratos, mg N-NO'+'\u2083'+' L'+ r'$^{-1}$', fontsize=15)
#plt.legend(loc='lower left', fontsize=10)
plt.show()

plt.figure()
#plt.plot(tiempo_datos, nitrate_datos, 'o', color='green', ms=5, label='Datos experimentales')
plt.plot([i for sublist in tiempo for i in sublist], [i for sublist in dbp for i in sublist], '-', color='orange', lw=2.5, label='MAn2')
plt.xlabel('Tiempo, d', fontsize=15)
#plt.ylim(0.2, 0.4)
plt.ylabel('Nitratos, mg N-NO'+'\u2083'+' L' + r'$^{-1}$', fontsize=15)
#plt.legend(loc='lower left', fontsize=10)
plt.show()