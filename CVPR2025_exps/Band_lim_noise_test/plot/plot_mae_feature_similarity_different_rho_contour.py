import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm

ppd = 60

save_root_path = 'contour_plots/mae/different_rho'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'../test/new_data_logs/mae/different_rho/mae_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_rho_sensitivity_data_band_lim_noise.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_rho_list = castleCSF_result_data['rho_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

all_mae_model_list = plot_json_data['mae_model_name']
plot_rho_matrix_list = plot_json_data['rho_matrix']
plot_contrast_matrix_list = plot_json_data['contrast_matrix']
plot_final_feature_L1_similarity_matrix_list = plot_json_data['final_feature_L1_similarity_matrix']
plot_final_feature_L2_similarity_matrix_list = plot_json_data['final_feature_L2_similarity_matrix']
plot_final_feature_cos_similarity_matrix_list = plot_json_data['final_feature_cos_similarity_matrix']
plot_final_feature_L1_similarity_matrix_f_list = plot_json_data['final_feature_L1_similarity_matrix_fix_random_seed']
plot_final_feature_L2_similarity_matrix_f_list = plot_json_data['final_feature_L2_similarity_matrix_fix_random_seed']
plot_final_feature_cos_similarity_matrix_f_list = plot_json_data['final_feature_cos_similarity_matrix_fix_random_seed']
x_rho_ticks = [0.5, 1, 2, 4, 8, 16, 32]
y_contrast_ticks = [0.001, 0.01, 0.1, 1]
y_sensitivity_ticks = [1, 10, 100, 1000]

for mae_model_index in tqdm(range(len(all_mae_model_list))):
    mae_model_name = all_mae_model_list[mae_model_index]
    real_save_path = os.path.join(save_root_path, mae_model_name)
    os.makedirs(real_save_path, exist_ok=True)
    plot_rho_matrix = np.array(plot_rho_matrix_list[mae_model_index])
    plot_contrast_matrix = np.array(plot_contrast_matrix_list[mae_model_index])

    plot_final_feature_L1_similarity_matrix = np.array(plot_final_feature_L1_similarity_matrix_list[mae_model_index])
    plot_final_feature_L2_similarity_matrix = np.array(plot_final_feature_L2_similarity_matrix_list[mae_model_index])
    plot_final_feature_cos_similarity_matrix = np.array(plot_final_feature_cos_similarity_matrix_list[mae_model_index])
    plot_final_feature_L1_similarity_matrix_f = np.array(plot_final_feature_L1_similarity_matrix_f_list[mae_model_index])
    plot_final_feature_L2_similarity_matrix_f = np.array(plot_final_feature_L2_similarity_matrix_f_list[mae_model_index])
    plot_final_feature_cos_similarity_matrix_f = np.array(
        plot_final_feature_cos_similarity_matrix_f_list[mae_model_index])
    plot_figure_data_matrix_list = [plot_final_feature_L1_similarity_matrix,
                                    plot_final_feature_L2_similarity_matrix,
                                    plot_final_feature_cos_similarity_matrix,
                                    plot_final_feature_L1_similarity_matrix_f,
                                    plot_final_feature_L2_similarity_matrix_f,
                                    plot_final_feature_cos_similarity_matrix_f]
    plot_figure_name_list = [f'{mae_model_name} - L1 similarity',
                             f'{mae_model_name} - L2 similarity',
                             f'{mae_model_name} - cos similarity',
                             f'{mae_model_name} - L1 similarity - fix_random_seed',
                             f'{mae_model_name} - L2 similarity - fix_random_seed',
                             f'{mae_model_name} - cos similarity - fix_random_seed']

    for figure_index in range(len(plot_figure_name_list)):
        plt.figure(figsize=(5, 3.5), dpi=300)
        plt.contourf(plot_rho_matrix, 1 / plot_contrast_matrix, plot_figure_data_matrix_list[figure_index],
                     levels=20, cmap='rainbow', alpha=0.3)
        plt.contour(plot_rho_matrix, 1 / plot_contrast_matrix, plot_figure_data_matrix_list[figure_index],
                    levels=20, cmap='rainbow')
        plt.plot(castleCSF_result_rho_list, castleCSF_result_sensitivity_list, 'k', linestyle='--', linewidth=2,
                 label='castleCSF prediction')
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
        plt.savefig(os.path.join(real_save_path, plot_figure_name_list[figure_index]),
                    dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()


