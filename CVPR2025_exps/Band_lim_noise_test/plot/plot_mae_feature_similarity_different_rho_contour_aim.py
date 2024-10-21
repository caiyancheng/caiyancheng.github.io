import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import root_mean_squared_error
ppd = 60
from scipy.stats import pearsonr, spearmanr

save_root_path = 'new_contour_plots/mae/different_rho'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'../test/new_data_logs/mae/different_rho/mae_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_{ppd}_final_aim.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
aim_cos_similarity_nf_list = plot_json_data['aim_cos_similarity_list']
aim_cos_similarity_f_list = plot_json_data['aim_cos_similarity_f_list']
rho_list = plot_json_data['rho_list']
multiplier_list = plot_json_data['multiplier_list']
Spearman_matrix_cos_nf_list = plot_json_data['Spearman_matrix_cos_list']
Spearman_matrix_cos_f_list = plot_json_data['Spearman_matrix_cos_f_list']

castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_rho_sensitivity_data_band_lim_noise.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_rho_list = castleCSF_result_data['rho_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

backbone_name_list = plot_json_data['mae_model_name']
plot_rho_matrix_list = plot_json_data['rho_matrix']
plot_contrast_matrix_list = plot_json_data['contrast_matrix']
plot_final_feature_cos_similarity_matrix_list = plot_json_data['final_feature_cos_similarity_matrix']
plot_final_feature_cos_similarity_matrix_f_list = plot_json_data['final_feature_cos_similarity_matrix_fix_random_seed']
x_rho_ticks = [0.5, 1, 2, 4, 8, 16, 32]
y_contrast_ticks = [0.001, 0.01, 0.1, 1]
y_sensitivity_ticks = [1, 10, 100, 1000]

for backbone_index in tqdm(range(len(backbone_name_list))):
    backbone_name = backbone_name_list[backbone_index]
    real_save_path = os.path.join(save_root_path, backbone_name)
    os.makedirs(real_save_path, exist_ok=True)
    aim_cos_similarity_nf = aim_cos_similarity_nf_list[backbone_index]
    aim_cos_similarity_f = aim_cos_similarity_f_list[backbone_index]
    Spearman_matrix_cos_nf = Spearman_matrix_cos_nf_list[backbone_index]
    Spearman_matrix_cos_f = Spearman_matrix_cos_f_list[backbone_index]

    plot_rho_matrix = np.array(plot_rho_matrix_list[backbone_index])
    plot_contrast_matrix = np.array(plot_contrast_matrix_list[backbone_index])

    plot_final_feature_cos_similarity_matrix = np.array(plot_final_feature_cos_similarity_matrix_list[backbone_index])
    plot_final_feature_cos_similarity_matrix_f = np.array(
        plot_final_feature_cos_similarity_matrix_f_list[backbone_index])

    aim_cos_similarity_list = [aim_cos_similarity_nf, aim_cos_similarity_f]
    Spearman_matrix_cos_list = [Spearman_matrix_cos_nf, Spearman_matrix_cos_f]
    plot_cos_similarity_matrix_list = [plot_final_feature_cos_similarity_matrix, plot_final_feature_cos_similarity_matrix_f]
    plot_figure_name_list = ['cos similarity - mae', 'cos similarity - mae fix_random_seed']

    for figure_index in range(len(plot_figure_name_list)):
        Spearman_matrix_cos = Spearman_matrix_cos_list[figure_index]
        X_Spearman_multiplier = []
        Y_Spearman_Score = []
        for area_index in range(len(Spearman_matrix_cos)):
            X_Spearman_multiplier += multiplier_list
            Y_Spearman_Score += Spearman_matrix_cos[area_index]
        correlation, p_value = spearmanr(X_Spearman_multiplier, Y_Spearman_Score)
        print(backbone_name)
        print(plot_figure_name_list[figure_index])
        print("Spearman Correlation:", round(correlation, 4))

        plt.figure(figsize=(5, 3.5), dpi=300)
        plt.contourf(plot_rho_matrix, 1 / plot_contrast_matrix, plot_cos_similarity_matrix_list [figure_index],
                     levels=20, cmap='rainbow', alpha=0.3)
        contours = plt.contour(plot_rho_matrix, 1 / plot_contrast_matrix, plot_cos_similarity_matrix_list [figure_index],
                    levels=20, cmap='rainbow')

        diff_list = list(abs(contours.levels - aim_cos_similarity_list[figure_index]))
        min_diff_value = min(diff_list)
        min_diff_index = diff_list.index(min_diff_value)
        min_diff_level_value = contours.levels[min_diff_index]
        while min_diff_level_value >= 1:
            min_diff_index = min_diff_index - 1
            min_diff_level_value = contours.levels[min_diff_index]
        index = list(contours.levels).index(min_diff_level_value)
        collection = contours.collections[index]

        min_Y_rho_list = []
        for rho_index, rho_value in enumerate(rho_list):
            y_values_at_target_x = []
            for path in collection.get_paths():
                vertices = path.vertices
                x_values = vertices[:, 0]
                y_values = vertices[:, 1]

                if x_values.min() <= rho_value <= x_values.max():
                    f_interp = interp1d(x_values, y_values, kind='linear', bounds_error=False, fill_value="extrapolate")
                    y_at_target_x = f_interp(rho_value)
                    y_values_at_target_x.append(y_at_target_x)
            if y_values_at_target_x:
                min_y_value = float(min(y_values_at_target_x))
            else:
                min_y_value = 1
            min_Y_rho_list.append(min_y_value)
        min_Y_rho_array = np.array(min_Y_rho_list)
        gt_f_interp = interp1d(castleCSF_result_rho_list, castleCSF_result_sensitivity_list, kind='linear',
                               fill_value="extrapolate")
        gt_Y_array = gt_f_interp(rho_list)
        log_RMSE_loss = root_mean_squared_error(np.log10(min_Y_rho_array), np.log10(gt_Y_array))
        print('RMSE: ', round(log_RMSE_loss, 4))
        correlation, p_value = pearsonr(np.log10(min_Y_rho_array), np.log10(gt_Y_array))
        print('Pearson: ', round(correlation, 4))

        plt.plot(castleCSF_result_rho_list, castleCSF_result_sensitivity_list, 'k', linestyle='--', linewidth=2,
                 label='castleCSF')
        plt.xlabel('Spatial Frequency (cpd)', fontsize=12)
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

