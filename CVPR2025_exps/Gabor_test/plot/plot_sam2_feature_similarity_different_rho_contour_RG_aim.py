import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr, spearmanr
ppd_list = [60] #[30, 35, 40, 45, 50, 55, 60]

save_root_path = 'new_contour_plots/sam2/different_rho_RG'
os.makedirs(save_root_path, exist_ok=True)

for ppd_index in range(len(ppd_list)):
    ppd_number = ppd_list[ppd_index]
    json_data_path = rf'../test/new_data_logs/sam2/different_rho_RG/sam2_test_on_gabors_different_rho_contour_plot_ppd_{ppd_number}_RG_final_aim.json'
    with open(json_data_path, 'r') as fp:
        plot_json_data = json.load(fp)
    aim_cos_similarity_list = plot_json_data['aim_cos_similarity_list']
    rho_list = plot_json_data['rho_RG_list']
    multiplier_list = plot_json_data['multiplier_list']
    Spearman_matrix_cos_list = plot_json_data['Spearman_matrix_cos_list']

    castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_rho_sensitivity_data_RG.json'
    with open(castleCSF_result_json, 'r') as fp:
        castleCSF_result_data = json.load(fp)
    castleCSF_result_rho_list = castleCSF_result_data['rho_list']
    castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

    sam_model_name_list = plot_json_data['sam_model_name']
    plot_rho_matrix_list = plot_json_data['rho_matrix']
    plot_contrast_matrix_list = plot_json_data['contrast_matrix']
    plot_final_feature_cos_similarity_matrix_list = plot_json_data['final_feature_cos_similarity_matrix']
    x_rho_ticks = [0.5, 1, 2, 4, 8, 16, 32]
    y_contrast_ticks = [0.001, 0.01, 0.1, 1]
    y_sensitivity_ticks = [5, 10, 100, 1000]

    for sam_model_index in tqdm(range(len(sam_model_name_list))):
        sam_model_name = sam_model_name_list[sam_model_index]
        sam_model_name_simple = sam_model_name.split('/')[-1]
        real_save_path = os.path.join(save_root_path, f'ppd_{ppd_number}', sam_model_name_simple)
        os.makedirs(real_save_path, exist_ok=True)

        Spearman_matrix_cos = Spearman_matrix_cos_list[sam_model_index]
        X_Spearman_multiplier = []
        Y_Spearman_Score = []
        for area_index in range(len(Spearman_matrix_cos)):
            X_Spearman_multiplier += multiplier_list
            Y_Spearman_Score += Spearman_matrix_cos[area_index]
        correlation, p_value = spearmanr(X_Spearman_multiplier, Y_Spearman_Score)
        print(sam_model_name)
        print("Spearman Correlation:", round(correlation, 4))

        plot_rho_matrix = np.array(plot_rho_matrix_list[sam_model_index])
        plot_contrast_matrix = np.array(plot_contrast_matrix_list[sam_model_index])
        plot_final_feature_cos_similarity_matrix = np.array(
            plot_final_feature_cos_similarity_matrix_list[sam_model_index])

        plt.figure(figsize=(5, 3.5), dpi=300)
        plt.contourf(plot_rho_matrix, 1 / plot_contrast_matrix, plot_final_feature_cos_similarity_matrix,
                     levels=20, cmap='rainbow', alpha=0.3)
        contours = plt.contour(plot_rho_matrix, 1 / plot_contrast_matrix, plot_final_feature_cos_similarity_matrix,
                    levels=20, cmap='rainbow')

        aim_cos_similarity = aim_cos_similarity_list[sam_model_index]
        diff_list = list(abs(contours.levels - aim_cos_similarity))
        min_diff_value = min(diff_list)
        min_diff_index = diff_list.index(min_diff_value)
        min_diff_level_value = contours.levels[min_diff_index]
        while min_diff_level_value >= 1:
            min_diff_index = min_diff_index - 1
            min_diff_level_value = contours.levels[min_diff_index]
        index = list(contours.levels).index(min_diff_level_value)
        collection = contours.collections[index]

        index = list(contours.levels).index(min_diff_level_value)
        collection = contours.collections[index]

        min_Y_rho_list = []
        for rho_index, rho_value in enumerate(rho_list):
            y_values_at_target_x = []
            for path in collection.get_paths():
                vertices = path.vertices
                x_values = vertices[:, 0]
                y_values = vertices[:, 1]

                unique_x_values = np.unique(x_values)
                min_y_values = np.array([y_values[x_values == x].min() for x in unique_x_values])
                x_values = unique_x_values
                y_values = min_y_values

                if x_values.min() <= rho_value <= x_values.max():
                    f_interp = interp1d(x_values, y_values, kind='linear', bounds_error=False, fill_value="extrapolate")
                    y_at_target_x = f_interp(rho_value)
                    y_values_at_target_x.append(y_at_target_x)
                elif x_values.min() > rho_value:
                    y_values_at_target_x.append(y_values[0])
                elif x_values.max() < rho_value:
                    y_values_at_target_x.append(y_values[-1])
            min_y_value = float(min(y_values_at_target_x))
            min_Y_rho_list.append(min_y_value)
        min_Y_rho_array = np.array(min_Y_rho_list)
        gt_f_interp = interp1d(castleCSF_result_rho_list, castleCSF_result_sensitivity_list, kind='linear',
                               fill_value="extrapolate")
        gt_Y_array = gt_f_interp(rho_list)
        log_RMSE_loss = root_mean_squared_error(np.log10(min_Y_rho_array), np.log10(gt_Y_array))
        print(sam_model_name)
        print('RMSE: ', round(log_RMSE_loss, 4))
        correlation, p_value = pearsonr(np.log10(min_Y_rho_array), np.log10(gt_Y_array))
        print('Pearson: ', round(correlation, 4))

        plt.plot(castleCSF_result_rho_list, castleCSF_result_sensitivity_list, 'k', linestyle='--', linewidth=2,
                 label='castleCSF (RG)')
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
        plt.savefig(os.path.join(real_save_path, f'cos similarity - sam2 - ppd_{ppd_number}.png'),
                    dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
