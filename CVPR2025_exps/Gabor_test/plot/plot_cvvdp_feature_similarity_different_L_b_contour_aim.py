import sys

sys.path.append('E:\Py_codes\LVM_Comparision')
import matplotlib.pyplot as plt
import json
import numpy as np
import math
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import root_mean_squared_error
ppd = 60
from scipy.stats import pearsonr, spearmanr

save_root_path = 'new_contour_plots/cvvdp/different_luminance'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'../test/new_data_logs/cvvdp/different_luminance/cvvdp_test_on_gabors_different_L_b_contour_plot_ppd_{ppd}_final_aim.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
aim_JOD = plot_json_data['aim_JOD']
L_list = plot_json_data['L_list']
multiplier_list = plot_json_data['multiplier_list']
Spearman_matrix_JOD = plot_json_data['Spearman_matrix_JOD']

castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_luminance_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_luminance_list = castleCSF_result_data['luminance_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

plot_L_b_matrix_list = plot_json_data['L_b_matrix']
plot_contrast_matrix_list = plot_json_data['contrast_matrix']
plot_JOD_score_matrix_list = plot_json_data['JOD_score_matrix']
x_luminance_ticks = [0.1, 1, 10, 100]
y_contrast_ticks = [0.001, 0.01, 0.1, 1]
y_sensitivity_ticks = [1, 10, 100, 1000]

plot_L_b_matrix = np.array(plot_L_b_matrix_list[0])
plot_contrast_matrix = np.array(plot_contrast_matrix_list[0])
plot_JOD_score_matrix = np.array(plot_JOD_score_matrix_list[0])

X_Spearman_multiplier = []
Y_Spearman_Score = []
for L_index in range(len(Spearman_matrix_JOD)):
    X_Spearman_multiplier += multiplier_list
    Y_Spearman_Score += Spearman_matrix_JOD[L_index]
correlation, p_value = spearmanr(X_Spearman_multiplier, Y_Spearman_Score)
print("Spearman Correlation:", round(correlation, 4))

plt.figure(figsize=(5, 3.5), dpi=300)
plt.contourf(plot_L_b_matrix, 1 / plot_contrast_matrix, plot_JOD_score_matrix,
             levels=20, cmap='rainbow', alpha=0.3)
contours = plt.contour(plot_L_b_matrix, 1 / plot_contrast_matrix, plot_JOD_score_matrix,
            levels=20, cmap='rainbow')

diff_list = list(abs(contours.levels - aim_JOD))
min_diff_value = min(diff_list)
min_diff_index = diff_list.index(min_diff_value)
min_diff_level_value = contours.levels[min_diff_index]
index = list(contours.levels).index(min_diff_level_value)
collection = contours.collections[index]
while len(collection.get_paths()) == 0:
    min_diff_index = min_diff_index - 1
    min_diff_level_value = contours.levels[min_diff_index]
    index = list(contours.levels).index(min_diff_level_value)
    collection = contours.collections[index]

min_Y_L_list = []
for L_index, L_value in enumerate(L_list):
    y_values_at_target_x = []
    for path in collection.get_paths():
        vertices = path.vertices
        x_values = vertices[:, 0]
        y_values = vertices[:, 1]

        if x_values.min() <= L_value <= x_values.max():
            f_interp = interp1d(x_values, y_values, kind='linear', bounds_error=False, fill_value="extrapolate")
            y_at_target_x = f_interp(L_value)
            y_values_at_target_x.append(y_at_target_x)
        elif x_values.min() > L_value:
            y_values_at_target_x.append(y_values[0])
        elif x_values.max() < L_value:
            y_values_at_target_x.append(y_values[-1])
    min_y_value = float(min(y_values_at_target_x))

    min_Y_L_list.append(min_y_value)
min_Y_L_array = np.array(min_Y_L_list)
gt_f_interp = interp1d(castleCSF_result_luminance_list, castleCSF_result_sensitivity_list, kind='linear',
                       fill_value="extrapolate")
gt_Y_array = gt_f_interp(L_list)
log_RMSE_loss = root_mean_squared_error(np.log10(min_Y_L_array), np.log10(gt_Y_array))
print('RMSE: ', round(log_RMSE_loss, 4))
correlation, p_value = pearsonr(np.log10(min_Y_L_array), np.log10(gt_Y_array))
print('Pearson: ', round(correlation, 4))


plt.plot(castleCSF_result_luminance_list, castleCSF_result_sensitivity_list, 'k', linestyle='--', linewidth=2,
         label='castleCSF')
plt.xlabel('Luminance (cd/m$^2$)', fontsize=12)
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
plt.savefig(os.path.join(save_root_path, 'JOD Score - cvvdp'),
            dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()
