import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm

ppd = 60

save_root_path = 'new_contour_plots/baseline/different_c'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'new_data_logs/baseline/different_c/baseline_test_on_contrast_masking_different_c_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
foley_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/foley_contrast_masking_data_gabor.json'
with open(foley_result_json, 'r') as fp:
    foley_result_data = json.load(fp)
foley_result_x_mask_contrast_list = foley_result_data['mask_contrast_list']
foley_result_y_test_contrast_list = foley_result_data['test_contrast_list']

plot_contrast_mask_matrix_list = plot_json_data['contrast_mask_matrix']
plot_contrast_test_matrix_list = plot_json_data['contrast_test_matrix']
plot_L1_similarity_matrix_list = plot_json_data['L1_similarity_matrix']
plot_L2_similarity_matrix_list = plot_json_data['L2_similarity_matrix']
plot_cos_similarity_matrix_list = plot_json_data['cos_similarity_matrix']
x_contrast_mask_ticks = [0.01, 0.1]
y_contrast_test_ticks = [0.01, 0.1]

plot_contrast_mask_matrix = np.array(plot_contrast_mask_matrix_list[0])
plot_contrast_test_matrix = np.array(plot_contrast_test_matrix_list[0])
plot_L1_similarity_matrix = np.array(plot_L1_similarity_matrix_list[0])
plot_L2_similarity_matrix = np.array(plot_L2_similarity_matrix_list[0])
plot_cos_similarity_matrix = np.array(plot_cos_similarity_matrix_list[0])

plot_figure_data_matrix_list = [plot_L1_similarity_matrix,
                                plot_L2_similarity_matrix,
                                plot_cos_similarity_matrix]
plot_figure_name_list = [f'L1 similarity - baseline',
                         f'L2 similarity - baseline',
                         f'cos similarity - baseline']

for figure_index in range(len(plot_figure_name_list)):
    plt.figure(figsize=(5, 5))
    plt.contour(plot_contrast_mask_matrix, plot_contrast_test_matrix, plot_figure_data_matrix_list[figure_index],
                levels=20)
    plt.plot(foley_result_x_mask_contrast_list, foley_result_y_test_contrast_list, 'r', linestyle='--', linewidth=2,
             label='Human Results')
    plt.xlabel('Mask Contrast', fontsize=12)
    plt.ylabel('Test Contrast', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(x_contrast_mask_ticks, x_contrast_mask_ticks)
    plt.yticks(y_contrast_test_ticks, y_contrast_test_ticks)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(save_root_path, plot_figure_name_list[figure_index]))
    plt.close()

