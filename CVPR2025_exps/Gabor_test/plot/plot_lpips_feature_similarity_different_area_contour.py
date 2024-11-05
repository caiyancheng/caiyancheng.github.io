import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import matplotlib.pyplot as plt
import json
import numpy as np
import math
import os
ppd = 60

save_root_path = 'contour_plots/lpips/different_area'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'../test/new_data_logs/lpips/different_area/lpips_test_on_gabors_different_area_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_area_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_area_list = castleCSF_result_data['area_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

plot_area_matrix = np.array(plot_json_data['area_matrix'])[0]
plot_contrast_matrix = np.array(plot_json_data['contrast_matrix'])[0]
plot_loss_fn_alex_matrix = np.array(plot_json_data['loss_fn_alex_matrix'])[0]
plot_loss_fn_vgg_matrix = np.array(plot_json_data['loss_fn_vgg_matrix'])[0]
x_area_ticks = [0.1, 1]
y_contrast_ticks = [0.001, 0.01, 0.1, 1]
y_sensitivity_ticks = [1, 10, 100, 1000]

plot_figure_data_matrix_list = [plot_loss_fn_alex_matrix, plot_loss_fn_vgg_matrix]
plot_figure_name_list = ['LPIPS - AlexNet', 'LPIPS - VggNet']

for figure_index in range(len(plot_figure_name_list)):
    plt.figure(figsize=(5, 3.5), dpi=300)
    plt.contourf(plot_area_matrix, 1 / plot_contrast_matrix, plot_figure_data_matrix_list[figure_index],
                 levels=20, cmap='rainbow', alpha=0.3)
    plt.contour(plot_area_matrix, 1 / plot_contrast_matrix, plot_figure_data_matrix_list[figure_index],
                levels=20, cmap='rainbow')
    plt.plot(castleCSF_result_area_list, castleCSF_result_sensitivity_list, 'k', linestyle='--', linewidth=2,
             label='castleCSF prediction')
    plt.xlabel('Stimulus Area (degree$^2$)', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    # plt.xlim([math.pi * 0.1 ** 2, math.pi * 1 ** 2])
    plt.xlim([plot_area_matrix.min(), plot_area_matrix.max()])
    plt.ylim([min(y_sensitivity_ticks), max(y_sensitivity_ticks)])
    plt.xticks(x_area_ticks, x_area_ticks)
    plt.yticks(y_sensitivity_ticks, y_sensitivity_ticks)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(save_root_path, plot_figure_name_list[figure_index]),
                dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

