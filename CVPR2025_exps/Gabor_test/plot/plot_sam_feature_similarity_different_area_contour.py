import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import matplotlib.pyplot as plt
import json
import numpy as np
import math
import os
from tqdm import tqdm

ppd_list = [60] #[30, 35, 40, 45, 50, 55, 60]

save_root_path = 'contour_plots_arc_scale/sam/different_area'
os.makedirs(save_root_path, exist_ok=True)

for ppd_index in range(len(ppd_list)):
    ppd_number = ppd_list[ppd_index]
    json_data_path = rf'../test/new_data_logs/sam/different_area/sam_test_on_gabors_different_area_contour_plot_ppd_{ppd_number}_final.json'
    with open(json_data_path, 'r') as fp:
        plot_json_data = json.load(fp)
    castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_area_sensitivity_data.json'
    with open(castleCSF_result_json, 'r') as fp:
        castleCSF_result_data = json.load(fp)
    castleCSF_result_area_list = castleCSF_result_data['area_list']
    castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

    sam_model_name_list = plot_json_data['sam_model_name']
    plot_radius_matrix_list = plot_json_data['radius_matrix']
    plot_area_matrix_list = plot_json_data['area_matrix']
    plot_contrast_matrix_list = plot_json_data['contrast_matrix']
    plot_final_feature_L1_similarity_matrix_list = plot_json_data['final_feature_L1_similarity_matrix']
    plot_final_feature_L2_similarity_matrix_list = plot_json_data['final_feature_L2_similarity_matrix']
    plot_final_feature_cos_similarity_matrix_list = plot_json_data['final_feature_cos_similarity_matrix']
    x_area_ticks = [0.1, 1]
    y_contrast_ticks = [0.001, 0.01, 0.1, 1]
    y_sensitivity_ticks = [1, 10, 100, 1000]

    max_L1_L2_json_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Compute_Max_Loss_all_models/max_L1_L2_sam.json'
    with open(max_L1_L2_json_path, 'r') as fp:
        json_data = json.load(fp)

    for sam_model_index in tqdm(range(len(sam_model_name_list))):
        sam_model_name = sam_model_name_list[sam_model_index]
        sam_model_name_simple = sam_model_name.split('/')[-1]
        real_save_path = os.path.join(save_root_path, f'ppd_{ppd_number}', sam_model_name_simple)
        os.makedirs(real_save_path, exist_ok=True)
        plot_area_matrix = np.array(plot_area_matrix_list[sam_model_index])
        plot_contrast_matrix = np.array(plot_contrast_matrix_list[sam_model_index])
        plot_final_feature_L1_similarity_matrix = np.array(
            plot_final_feature_L1_similarity_matrix_list[sam_model_index])
        plot_final_feature_L2_similarity_matrix = np.array(
            plot_final_feature_L2_similarity_matrix_list[sam_model_index])
        plot_final_feature_cos_similarity_matrix = np.array(
            plot_final_feature_cos_similarity_matrix_list[sam_model_index])

        plot_final_feature_cos_similarity_matrix[plot_final_feature_cos_similarity_matrix > 1] = 1
        plot_figure_data_matrix_list = [plot_final_feature_L1_similarity_matrix / json_data['max_L1'][sam_model_index],
                                        plot_final_feature_L2_similarity_matrix / json_data['max_L2'][sam_model_index],
                                        np.arccos(plot_final_feature_cos_similarity_matrix) / np.arccos(-1),
                                        np.arccos(plot_final_feature_cos_similarity_matrix) / np.arccos(
                                            json_data['min_cos'][sam_model_index])]
        plot_figure_name_list = [f'{sam_model_name} - L1 similarity - final feature',
                                 f'{sam_model_name} - L2 similarity - final feature',
                                 f'{sam_model_name} - arccos cos similarity - final feature',
                                 f'{sam_model_name} - arccos-scale cos similarity - final feature']

        for figure_index in range(len(plot_figure_name_list)):
            plt.figure(figsize=(5, 3), dpi=300)
            levels = np.linspace(0, 1, 50)
            plt.contourf(plot_area_matrix, 1 / plot_contrast_matrix, plot_figure_data_matrix_list[figure_index],
                         levels=levels, cmap='rainbow', alpha=0.3)
            plt.contour(plot_area_matrix, 1 / plot_contrast_matrix, plot_figure_data_matrix_list[figure_index],
                        levels=levels, cmap='rainbow')
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
            plt.savefig(os.path.join(real_save_path, plot_figure_name_list[figure_index] + f'_ppd_{ppd_number}.png'),
                        dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close()