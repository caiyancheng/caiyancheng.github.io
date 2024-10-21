import sys

sys.path.append('E:\Py_codes\LVM_Comparision')
import matplotlib.pyplot as plt
import json
import numpy as np
import math
import os
from tqdm import tqdm

ppd = 60

save_root_path = 'new_contour_plots/baseline/different_luminance'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'../test/new_data_logs/baseline/different_luminance/baseline_test_on_gabors_different_L_b_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
optimize_x_luminance = plot_json_data['optimize_x_luminance']
optimize_y_sensitivity = plot_json_data['optimize_y_sensitivity']

castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_luminance_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_luminance_list = castleCSF_result_data['luminance_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

plot_L_b_matrix_list = plot_json_data['L_b_matrix']
plot_contrast_matrix_list = plot_json_data['contrast_matrix']
plot_cos_similarity_matrix_list = plot_json_data['cos_similarity_matrix']
x_luminance_ticks = [0.1, 1, 10, 100]
y_contrast_ticks = [0.001, 0.01, 0.1, 1]
y_sensitivity_ticks = [1, 10, 100, 1000]

plot_L_b_matrix = np.array(plot_L_b_matrix_list[0])
plot_contrast_matrix = np.array(plot_contrast_matrix_list[0])
plot_cos_similarity_matrix = np.array(plot_cos_similarity_matrix_list[0])

plt.figure(figsize=(5, 3.5), dpi=300)
plt.contourf(plot_L_b_matrix, 1 / plot_contrast_matrix, plot_cos_similarity_matrix,
             levels=20, cmap='rainbow', alpha=0.3)
plt.contour(plot_L_b_matrix, 1 / plot_contrast_matrix, plot_cos_similarity_matrix,
            levels=20, cmap='rainbow')
plt.plot(castleCSF_result_luminance_list, castleCSF_result_sensitivity_list, 'k', linestyle='--', linewidth=2,
         label='castleCSF')
plt.plot(optimize_x_luminance, optimize_y_sensitivity, 'k', linestyle='-', linewidth=2,
         label='model')
plt.xlabel('Stimulus Luminance (cd/m$^2$)', fontsize=12)
plt.ylabel('Sensitivity', fontsize=12)
plt.xscale('log')
plt.yscale('log')
# plt.xlim([0.1, 200])
plt.xlim([plot_L_b_matrix.min(), plot_L_b_matrix.max()])
plt.ylim([min(y_sensitivity_ticks), max(y_sensitivity_ticks)])
plt.xticks(x_luminance_ticks, x_luminance_ticks)
plt.yticks(y_sensitivity_ticks, y_sensitivity_ticks)
# plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(save_root_path, 'cos similarity - baseline'),
            dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()
