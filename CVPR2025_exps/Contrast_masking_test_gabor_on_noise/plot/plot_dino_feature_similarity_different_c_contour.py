import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm

ppd = 60

save_root_path = 'new_contour_plots/dino/different_c'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'new_data_logs/dino/different_c/dino_test_on_contrast_masking_different_c_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
foley_result_json = r'contrast_masking_data_gabor_on_noise.json'
with open(foley_result_json, 'r') as fp:
    foley_result_data = json.load(fp)
foley_result_x_mask_contrast_list = foley_result_data['mask_contrast_list']
foley_result_y_test_contrast_list = foley_result_data['test_contrast_list']

backbone_name_list = plot_json_data['backbone_name']
plot_contrast_mask_matrix_list = plot_json_data['contrast_mask_matrix']
plot_contrast_test_matrix_list = plot_json_data['contrast_test_matrix']
plot_final_feature_L1_similarity_matrix_list = plot_json_data['final_feature_L1_similarity_matrix']
plot_final_feature_L2_similarity_matrix_list = plot_json_data['final_feature_L2_similarity_matrix']
plot_final_feature_cos_similarity_matrix_list = plot_json_data['final_feature_cos_similarity_matrix']
plot_intermediate_feature_L1_similarity_matrix_list = plot_json_data['intermediate_feature_L1_similarity_matrix']
plot_intermediate_feature_L2_similarity_matrix_list = plot_json_data['intermediate_feature_L2_similarity_matrix']
plot_intermediate_feature_cos_similarity_matrix_list = plot_json_data['intermediate_feature_cos_similarity_matrix']
x_contrast_mask_ticks = [0.01, 0.1]
y_contrast_test_ticks = [0.01, 0.1]

for backbone_index in tqdm(range(len(backbone_name_list))):
    backbone_name = backbone_name_list[backbone_index]
    real_save_path = os.path.join(save_root_path, backbone_name)
    os.makedirs(real_save_path, exist_ok=True)

    plot_contrast_mask_matrix = np.array(plot_contrast_mask_matrix_list[backbone_index])
    plot_contrast_test_matrix = np.array(plot_contrast_test_matrix_list[backbone_index])
    plot_final_feature_L1_similarity_matrix = np.array(plot_final_feature_L1_similarity_matrix_list[backbone_index])
    plot_final_feature_L2_similarity_matrix = np.array(plot_final_feature_L2_similarity_matrix_list[backbone_index])
    plot_final_feature_cos_similarity_matrix = np.array(plot_final_feature_cos_similarity_matrix_list[backbone_index])
    if backbone_name.startswith('dino_vit'):
        plot_intermediate_feature_L1_similarity_matrix = np.array(
            plot_intermediate_feature_L1_similarity_matrix_list[backbone_index])
        plot_intermediate_feature_L2_similarity_matrix = np.array(
            plot_intermediate_feature_L2_similarity_matrix_list[backbone_index])
        plot_intermediate_feature_cos_similarity_matrix = np.array(
            plot_intermediate_feature_cos_similarity_matrix_list[backbone_index])
        plot_figure_data_matrix_list = [plot_final_feature_L1_similarity_matrix,
                                        plot_final_feature_L2_similarity_matrix,
                                        plot_final_feature_cos_similarity_matrix,
                                        plot_intermediate_feature_L1_similarity_matrix,
                                        plot_intermediate_feature_L2_similarity_matrix,
                                        plot_intermediate_feature_cos_similarity_matrix]
        plot_figure_name_list = [f'{backbone_name} - L1 similarity - final feature',
                                 f'{backbone_name} - L2 similarity - final feature',
                                 f'{backbone_name} - cos similarity - final feature',
                                 f'{backbone_name} - L1 similarity - intermediate feature',
                                 f'{backbone_name} - L2 similarity - intermediate feature',
                                 f'{backbone_name} - cos similarity - intermediate feature']
    else:
        plot_figure_data_matrix_list = [plot_final_feature_L1_similarity_matrix,
                                        plot_final_feature_L2_similarity_matrix,
                                        plot_final_feature_cos_similarity_matrix]
        plot_figure_name_list = [f'{backbone_name} - L1 similarity - final feature',
                                 f'{backbone_name} - L2 similarity - final feature',
                                 f'{backbone_name} - cos similarity - final feature']

    for figure_index in range(len(plot_figure_name_list)):
        plt.figure(figsize=(5, 4), dpi=300)
        # heatmap = plt.scatter(plot_contrast_mask_matrix.flatten(), plot_contrast_test_matrix.flatten(),
        #                       plot_figure_data_matrix_list[figure_index].flatten(), c=plot_figure_data_matrix_list[figure_index].flatten(), cmap='rainbow')
        plt.contourf(plot_contrast_mask_matrix, plot_contrast_test_matrix,
                     plot_figure_data_matrix_list[figure_index],
                     levels=20, cmap='rainbow', alpha=0.3)
        plt.contour(plot_contrast_mask_matrix, plot_contrast_test_matrix,
                    plot_figure_data_matrix_list[figure_index],
                    levels=20, cmap='rainbow')
        # plt.clabel(contour, inline=True, fontsize=8)
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
        plt.savefig(os.path.join(real_save_path, plot_figure_name_list[figure_index]), dpi=300, bbox_inches='tight',
                    pad_inches=0.02)
        plt.close()

