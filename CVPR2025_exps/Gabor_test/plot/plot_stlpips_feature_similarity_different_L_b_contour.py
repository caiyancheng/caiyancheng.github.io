import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
ppd = 60

save_root_path = 'contour_plots_arc_scale/stlpips/different_luminance'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'../test/new_data_logs/stlpips/different_luminance/stlpips_test_on_gabors_different_L_b_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_luminance_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_luminance_list = castleCSF_result_data['luminance_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

plot_L_b_matrix = np.array(plot_json_data['L_b_matrix'])[0]
plot_contrast_matrix = np.array(plot_json_data['contrast_matrix'])[0]
plot_loss_fn_alex_matrix = np.array(plot_json_data['loss_fn_alex_matrix'])[0]
plot_loss_fn_vgg_matrix = np.array(plot_json_data['loss_fn_vgg_matrix'])[0]
x_luminance_ticks = [0.1, 1, 10, 100]
y_contrast_ticks = [0.001, 0.01, 0.1, 1]
y_sensitivity_ticks = [1, 10, 100, 1000]

max_loss_json_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Compute_Max_Loss_all_models/max_loss_stlpips.json'
with open(max_loss_json_path, 'r') as fp:
    json_data = json.load(fp)
max_loss_alex = json_data['max_loss_alex']
max_loss_vgg = json_data['max_loss_vgg']

plot_figure_data_matrix_list = [plot_loss_fn_alex_matrix / max_loss_alex, plot_loss_fn_vgg_matrix / max_loss_vgg]
plot_figure_name_list = ['STLPIPS - AlexNet', 'STLPIPS - VggNet']

for figure_index in range(len(plot_figure_name_list)):
    plt.figure(figsize=(5, 3), dpi=300)
    levels = np.linspace(0, 1, 50)
    plt.contourf(plot_L_b_matrix, 1 / plot_contrast_matrix, plot_figure_data_matrix_list[figure_index],
                 levels=levels, cmap='rainbow', alpha=0.3)
    plt.contour(plot_L_b_matrix, 1 / plot_contrast_matrix, plot_figure_data_matrix_list[figure_index],
                levels=levels, cmap='rainbow')
    plt.plot(castleCSF_result_luminance_list, castleCSF_result_sensitivity_list, 'k', linestyle='--', linewidth=2,
             label='castleCSF prediction')
    plt.xlabel('Stimulus Luminance (cd/m$^2$)', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    # plt.xlim([0.1, 200])
    plt.xlim([plot_L_b_matrix.min(), plot_L_b_matrix.max()])
    plt.ylim([min(y_sensitivity_ticks), max(y_sensitivity_ticks)])
    plt.xticks(x_luminance_ticks, x_luminance_ticks)
    plt.yticks(y_sensitivity_ticks, y_sensitivity_ticks)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(save_root_path, plot_figure_name_list[figure_index]),
                dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

