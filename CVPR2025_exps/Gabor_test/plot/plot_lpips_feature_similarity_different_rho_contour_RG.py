import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import matplotlib.pyplot as plt
import json
import numpy as np
import math
import os

ppd = 60

save_root_path = 'new_contour_plots/lpips/different_rho_RG'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'new_data_logs/lpips/different_rho_RG/lpips_test_on_gabors_different_rho_contour_plot_ppd_{ppd}_RG_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_rho_sensitivity_data_RG.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_rho_list = castleCSF_result_data['rho_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

plot_rho_matrix = np.array(plot_json_data['rho_matrix'])[0]
plot_contrast_matrix = np.array(plot_json_data['contrast_matrix'])[0]
plot_loss_fn_alex_matrix = np.array(plot_json_data['loss_fn_alex_matrix'])[0]
plot_loss_fn_vgg_matrix = np.array(plot_json_data['loss_fn_vgg_matrix'])[0]
x_rho_ticks = [0.5, 1, 2, 4, 8, 16, 32]
y_contrast_ticks = [0.001, 0.01, 0.1, 1]
y_sensitivity_ticks = [5, 10, 100, 1000]

fig, axs = plt.subplots(2, 3, figsize=(15, 7))
plot_figure_data_matrix_list = [plot_loss_fn_alex_matrix, plot_loss_fn_vgg_matrix]
plot_figure_name_list = ['LPIPS - AlexNet', 'LPIPS - VggNet']

for figure_index in range(len(plot_figure_name_list)):
    plt.figure(figsize=(5, 3.5), dpi=300)
    plt.contourf(plot_rho_matrix, 1 / plot_contrast_matrix, plot_figure_data_matrix_list[figure_index],
                 levels=20, cmap='rainbow', alpha=0.3)
    plt.contour(plot_rho_matrix, 1 / plot_contrast_matrix, plot_figure_data_matrix_list[figure_index],
                levels=20, cmap='rainbow')
    plt.plot(castleCSF_result_rho_list, castleCSF_result_sensitivity_list, 'k', linestyle='--', linewidth=2,
             label='castleCSF prediction (RG)')
    plt.xlabel('Stimulus Spatial Frequency (cpd)', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    # plt.xlim([0.5, 32])
    plt.xlim([plot_rho_matrix.min(), plot_rho_matrix.max()])
    plt.ylim([min(y_sensitivity_ticks), max(y_sensitivity_ticks)])
    plt.xticks(x_rho_ticks, x_rho_ticks)
    plt.yticks(y_sensitivity_ticks, y_sensitivity_ticks)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(save_root_path, plot_figure_name_list[figure_index]),
                dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
