import sys

sys.path.append('E:\Py_codes\LVM_Comparision')
import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm
import math

ppd = 60

save_root_path = 'new_contour_plots/cvvdp/different_area'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'../test/new_data_logs/cvvdp/different_area/cvvdp_test_on_gabors_different_area_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
optimize_x_area = plot_json_data['optimize_x_area']
optimize_y_sensitivity = plot_json_data['optimize_y_sensitivity']

castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_area_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_area_list = castleCSF_result_data['area_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

plot_radius_matrix_list = plot_json_data['radius_matrix']
plot_area_matrix_list = plot_json_data['area_matrix']
plot_contrast_matrix_list = plot_json_data['contrast_matrix']
plot_cos_similarity_matrix_list = plot_json_data['JOD_score_matrix']
x_area_ticks = [0.1, 1]
y_contrast_ticks = [0.001, 0.01, 0.1, 1]
y_sensitivity_ticks = [1, 10, 100, 1000]

plot_area_matrix = np.array(plot_area_matrix_list[0])
plot_contrast_matrix = np.array(plot_contrast_matrix_list[0])
plot_cos_similarity_matrix = np.array(plot_cos_similarity_matrix_list[0])

plt.figure(figsize=(5, 3.5), dpi=300)
plt.contourf(plot_area_matrix, 1 / plot_contrast_matrix, plot_cos_similarity_matrix,
             levels=20, cmap='rainbow', alpha=0.3)
plt.contour(plot_area_matrix, 1 / plot_contrast_matrix, plot_cos_similarity_matrix,
            levels=20, cmap='rainbow')
plt.plot(castleCSF_result_area_list, castleCSF_result_sensitivity_list, 'k', linestyle='--', linewidth=2,
         label='castleCSF')
plt.plot(optimize_x_area, optimize_y_sensitivity, 'k', linestyle='-', linewidth=2,
         label='model')
plt.xlabel('Stimulus Area (degree$^2$)', fontsize=12)
plt.ylabel('Sensitivity', fontsize=12)
plt.xscale('log')
plt.yscale('log')
# plt.xlim([math.pi * 0.1 ** 2, math.pi * 1 ** 2])
plt.xlim([plot_area_matrix.min(), plot_area_matrix.max()])
plt.ylim([min(y_sensitivity_ticks), max(y_sensitivity_ticks)])
plt.xticks(x_area_ticks, x_area_ticks)
plt.yticks(y_sensitivity_ticks, y_sensitivity_ticks)
# plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(save_root_path, 'JOD - cvvdp'),
            dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()
