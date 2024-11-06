import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm

ppd = 60

save_root_path = 'contour_plots_arc_scale/vae/different_rho_RG'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'../test/new_data_logs/vae/different_rho_RG/vae_test_on_gabors_different_rho_contour_plot_ppd_{ppd}_RG_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_rho_sensitivity_data_RG.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_rho_list = castleCSF_result_data['rho_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

vae_model_name_list = plot_json_data['vae_model_name']
plot_rho_matrix_list = plot_json_data['rho_matrix']
plot_contrast_matrix_list = plot_json_data['contrast_matrix']
plot_final_feature_L1_similarity_matrix_list = plot_json_data['final_feature_L1_similarity_matrix']
plot_final_feature_L2_similarity_matrix_list = plot_json_data['final_feature_L2_similarity_matrix']
plot_final_feature_cos_similarity_matrix_list = plot_json_data['final_feature_cos_similarity_matrix']
x_rho_ticks = [0.5, 1, 2, 4, 8, 16, 32]
y_contrast_ticks = [0.001, 0.01, 0.1, 1]
y_sensitivity_ticks = [5, 10, 100, 1000]

max_L1_L2_json_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Compute_Max_Loss_all_models/max_L1_L2_vae.json'
with open(max_L1_L2_json_path, 'r') as fp:
    json_data = json.load(fp)

for vae_model_index in tqdm(range(len(vae_model_name_list))):
    vae_model_name = vae_model_name_list[vae_model_index]
    vae_model_name_simple = vae_model_name.split('/')[-1]
    real_save_path = os.path.join(save_root_path, vae_model_name_simple)
    os.makedirs(real_save_path, exist_ok=True)
    plot_rho_matrix = np.array(plot_rho_matrix_list[vae_model_index])
    plot_contrast_matrix = np.array(plot_contrast_matrix_list[vae_model_index])
    plot_final_feature_L1_similarity_matrix = np.array(plot_final_feature_L1_similarity_matrix_list[vae_model_index])
    plot_final_feature_L2_similarity_matrix = np.array(plot_final_feature_L2_similarity_matrix_list[vae_model_index])
    plot_final_feature_cos_similarity_matrix = np.array(plot_final_feature_cos_similarity_matrix_list[vae_model_index])

    plot_final_feature_cos_similarity_matrix[plot_final_feature_cos_similarity_matrix > 1] = 1
    plot_figure_data_matrix_list = [plot_final_feature_L1_similarity_matrix / json_data['max_L1'][vae_model_index],
                                    plot_final_feature_L2_similarity_matrix / json_data['max_L2'][vae_model_index],
                                    np.arccos(plot_final_feature_cos_similarity_matrix) / np.arccos(-1),
                                    np.arccos(plot_final_feature_cos_similarity_matrix) / np.arccos(
                                        json_data['min_cos'][vae_model_index])]
    plot_figure_name_list = [f'{vae_model_name_simple} - L1 similarity - final feature',
                             f'{vae_model_name_simple} - L2 similarity - final feature',
                             f'{vae_model_name_simple} - arccos cos similarity - final feature',
                             f'{vae_model_name_simple} - arccos-scale cos similarity - final feature']

    for figure_index in range(len(plot_figure_name_list)):
        plt.figure(figsize=(5, 3), dpi=300)
        levels = np.linspace(0, 1, 50)
        plt.contourf(plot_rho_matrix, 1 / plot_contrast_matrix, plot_figure_data_matrix_list[figure_index],
                     levels=levels, cmap='rainbow', alpha=0.3)
        plt.contour(plot_rho_matrix, 1 / plot_contrast_matrix, plot_figure_data_matrix_list[figure_index],
                    levels=levels, cmap='rainbow')
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
        plt.savefig(os.path.join(real_save_path, plot_figure_name_list[figure_index]+'.png'),
                    dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
