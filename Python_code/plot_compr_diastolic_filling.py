import pandas as pd
import matplotlib.pyplot as plt
import sys

# df_scaled_10 = pd.read_csv("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/COMSOL/EDP_10/P_volumes.csv")
# df_scaled_15 = pd.read_csv("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/COMSOL/EDP_15/P_volumes.csv")
# df_scaled_20 = pd.read_csv("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/COMSOL/EDP_20/P_volumes.csv")
# df_scaled_25 = pd.read_csv("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/COMSOL/EDP_25/P_volumes.csv")
# #df_scaled_45 = pd.read_csv("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/Diastolic_filling/P_volumes_scaled_45.0.csv")

# plt.figure()

# # Plotting Column1 vs. Column2
# plt.plot(df_scaled_10['bp_volume'], df_scaled_10['pressure'], marker='o', linestyle='-', color='b',alpha = 0.4, label = "EDP = 10mmHg")
# plt.plot(df_scaled_15['bp_volume'], df_scaled_15['pressure'], marker='o', linestyle='-', color='darkgreen',alpha = 0.4, label = "EDP = 15mmHg")
# plt.plot(df_scaled_20['bp_volume'], df_scaled_20['pressure'], marker='o', linestyle='-', color='darkred',alpha = 0.4, label = "EDP = 20mmHg")
# plt.plot(df_scaled_25['bp_volume'], df_scaled_25['pressure'], marker='o', linestyle='-', color='black',alpha = 0.4, label = "EDP = 25mmHg")
# #plt.plot(df_scaled_45['volume'], df_scaled_45['pressure_LV'][::10], marker='o', linestyle='-', color='purple',alpha = 0.4, label = "EDP = 45mmHg")

# # Adding labels and a title
# plt.xlabel('Volume [ml]', fontsize = 12)
# plt.ylabel('Pressure [mmHg]', fontsize = 12)
# plt.title('Diastolic filling predicted from COMSOL for scaled swine mechanical properties')
# plt.legend()
# plt.savefig("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/COMSOL/compr_diastolic_filling_scaled_swine_prams.png")

# # Show the plot
# plt.show()


# #create plot comparing diastolic filling for one EDP for the two mechanical properties
# df_scaled_15 = pd.read_csv("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/Diastolic_filling/P_volumes_scaled_25.0.csv")
# df_human_15 = pd.read_csv("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/Diastolic_filling/P_volumes_25.0.csv")

# plt.plot(df_human_15['volume'][::10], df_human_15['pressure_LV'][::10], marker='o', linestyle='-', color='b',alpha = 0.4, label = "human mechanical properties")
# plt.plot(df_scaled_15['volume'][::10], df_scaled_15['pressure_LV'][::10], marker='o', linestyle='-', color='darkred',alpha = 0.4, label = "scaled swine mechanical properties")

# plt.xlabel('Volume [ml]', fontsize = 12)
# plt.ylabel('Pressure [mmHg]', fontsize = 12)
# plt.title('Diastolic filling predicted by PINN for EDP = 25mmHg')
# plt.legend()
# plt.savefig("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/Diastolic_filling/diastolic_filling_dif_mech_properties_EDP25.png")

# # Show the plot
# plt.show()

# #create plot comparing diastolic filling for one EDP for the two mechanical properties
# df_PINN_400 = pd.read_csv("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/EDP_15/Epochs_400/P_volumes_meshcode.csv")
# df_PINN_500 = pd.read_csv("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/EDP_15/Epochs_500/P_volumes_meshcode.csv")
# df_comsol_template = pd.read_csv("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/COMSOL/EDP_15/EDP_15_template/P_volumes.csv")

# plt.plot(df_PINN_400['bp_volume'][:65], df_PINN_400['pressure'][:65], marker='o', linestyle='-', color='g',alpha = 0.4, label = "PINN--400 epochs 400 data points")
# plt.plot(df_PINN_400['bp_volume'][:65], df_PINN_500['pressure'][:65], marker='o', linestyle='-', color='red',alpha = 0.4, label = "PINN--500 epochs 400 data points")
# plt.plot(df_comsol_template['bp_volume'], df_comsol_template['pressure'], marker='o', linestyle='-', color='black',alpha = 0.4, label = "Comsol template mesh")


# plt.xlabel('Volume [ml]', fontsize = 12)
# plt.ylabel('Pressure [mmHg]', fontsize = 12)
# plt.title('Diastolic filling predicted for EDP = 10mmHg')
# plt.legend()

# #plt.savefig("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/COMSOL/EDP_10_compr_diastolic_filling.png")


# # Show the plot
# plt.show()

df_test = pd.read_csv("/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/Test/PINN_data/P_volumes.csv")

plt.plot(df_test['volume'], df_test['pressure_LV'], marker='o', linestyle='-', color='g',alpha = 0.4, label = "PINN--400 epochs 400 data points")

plt.show()