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

save_root_path = 'new_contour_plots/mae/different_c'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'../test/new_data_logs/mae/different_c/mae_test_on_contrast_masking_different_c_contour_plot_ppd_{ppd}_final_aim.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
aim_cos_similarity_list = plot_json_data['aim_cos_similarity_list']
valid_gt_indices = plot_json_data['valid_gt_indices']
gt_x_mask_C = plot_json_data['gt_x_mask_C']
gt_y_test_C = plot_json_data['gt_y_test_C']
multiplier_list = plot_json_data['multiplier_list']
Spearman_matrix_cos_list = plot_json_data['Spearman_matrix_cos_list']

foley_result_json = r'../contrast_masking_data_gabor_on_noise.json'
with open(foley_result_json, 'r') as fp:
    foley_result_data = json.load(fp)
foley_result_x_mask_contrast_list = foley_result_data['mask_contrast_list']
foley_result_y_test_contrast_list = foley_result_data['test_contrast_list']

backbone_name_list = plot_json_data['mae_model_name']
plot_contrast_mask_matrix_list = plot_json_data['contrast_mask_matrix']
plot_contrast_test_matrix_list = plot_json_data['contrast_test_matrix']
plot_final_feature_cos_similarity_matrix_list = plot_json_data['final_feature_cos_similarity_matrix']
x_contrast_mask_ticks = [0.01, 0.1]
y_contrast_test_ticks = [0.01, 0.1]

for backbone_index in tqdm(range(len(backbone_name_list))):
    backbone_name = backbone_name_list[backbone_index]
    real_save_path = os.path.join(save_root_path, backbone_name)
    os.makedirs(real_save_path, exist_ok=True)

    Spearman_matrix_cos = Spearman_matrix_cos_list[backbone_index]
    X_Spearman_multiplier = []
    Y_Spearman_Score = []
    for index in range(len(Spearman_matrix_cos)):
        X_Spearman_multiplier += multiplier_list
        Y_Spearman_Score += list(-np.array(Spearman_matrix_cos[index]))
    correlation, p_value = spearmanr(X_Spearman_multiplier, Y_Spearman_Score)
    print(backbone_name)
    print("Spearman Correlation:", round(correlation, 4))

    plot_contrast_mask_matrix = np.array(plot_contrast_mask_matrix_list[backbone_index])
    plot_contrast_test_matrix = np.array(plot_contrast_test_matrix_list[backbone_index])
    plot_final_feature_cos_similarity_matrix = np.array(plot_final_feature_cos_similarity_matrix_list[backbone_index])

    plt.figure(figsize=(5, 3.5), dpi=300)
    plt.contourf(plot_contrast_mask_matrix, plot_contrast_test_matrix,
                 plot_final_feature_cos_similarity_matrix,
                 levels=20, cmap='rainbow', alpha=0.3)
    contours = plt.contour(plot_contrast_mask_matrix, plot_contrast_test_matrix,
                plot_final_feature_cos_similarity_matrix,
                levels=20, cmap='rainbow')

    diff_list = list(abs(contours.levels - aim_cos_similarity_list[backbone_index]))
    min_diff_value = min(diff_list)
    min_diff_index = diff_list.index(min_diff_value)
    min_diff_level_value = contours.levels[min_diff_index]
    while min_diff_level_value >= 1:
        min_diff_index = min_diff_index - 1
        min_diff_level_value = contours.levels[min_diff_index]
    index = list(contours.levels).index(min_diff_level_value)
    collection = contours.collections[index]

    min_Y_Ct_list = []
    for Cm_index, Cm_value in enumerate(gt_x_mask_C):
        y_values_at_target_x = []
        for path in collection.get_paths():
            vertices = path.vertices
            x_values = vertices[:, 0]
            y_values = vertices[:, 1]

            if x_values.min() <= Cm_value <= x_values.max():
                f_interp = interp1d(x_values, y_values, kind='linear', bounds_error=False, fill_value="extrapolate")
                y_at_target_x = f_interp(Cm_value)
                y_values_at_target_x.append(y_at_target_x)
            elif x_values.min() > Cm_value:
                y_values_at_target_x.append(y_values.min())
            elif x_values.max() < Cm_value:
                y_values_at_target_x.append(y_values.min())
        min_y_value = float(min(y_values_at_target_x))
        min_Y_Ct_list.append(min_y_value)
    min_Y_Ct_array = np.array(min_Y_Ct_list)
    log_RMSE_loss = root_mean_squared_error(np.log10(min_Y_Ct_array), np.log10(gt_y_test_C))
    print('RMSE: ', round(log_RMSE_loss, 4))
    correlation, p_value = pearsonr(np.log10(min_Y_Ct_array), np.log10(gt_y_test_C))
    print('Pearson: ', round(correlation, 4))

    plt.plot(foley_result_x_mask_contrast_list, foley_result_y_test_contrast_list, 'k', linestyle='--', linewidth=2,
             label='Human Results', marker='o')
    plt.ylim([plot_contrast_test_matrix.min(), plot_contrast_test_matrix.max()])
    plt.xlim([plot_contrast_mask_matrix.min(), plot_contrast_mask_matrix.max()])
    plt.xlabel('Mask Contrast', fontsize=12)
    plt.ylabel('Test Contrast', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(x_contrast_mask_ticks, x_contrast_mask_ticks)
    plt.yticks(y_contrast_test_ticks, y_contrast_test_ticks)
    # plt.tight_layout()
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(os.path.join(real_save_path, 'cos similarity - mae'), dpi=300, bbox_inches='tight',
                pad_inches=0.02)
    plt.close()

