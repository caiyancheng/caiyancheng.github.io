import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm

ppd = 60

colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'indigo', 'violet']
save_root_path = 'new_contour_plots/mae/different_rho'
os.makedirs(save_root_path, exist_ok=True)
y_contrast_test_ticks = [0.001, 0.01, 0.1, 1]

json_data_path = fr'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Contrast_constancy\test\new_data_logs\mae\different_rho/mae_test_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
backbone_name_list = plot_json_data['backbone_name_list']
reference_contrast_list = plot_json_data['reference_contrast_list']
rho_test_list = plot_json_data['rho_test_list']
rho_gt_list = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]
rho_x_ticks = [0.25, 0.5, 1, 2, 4, 8, 16]

human_result_json_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Contrast_constancy/contrast_constancy_sin_5_cpd.json'
with open(human_result_json_path, 'r') as fp:
    human_result_data = json.load(fp)


for backbone_index in tqdm(range(len(backbone_name_list))):
    backbone_name = backbone_name_list[backbone_index]
    real_save_path = os.path.join(save_root_path, backbone_name)
    os.makedirs(real_save_path, exist_ok=True)
    plt.figure(figsize=(5, 4), dpi=300)
    legend_OK = 0
    for reference_contrast_index in range(len(reference_contrast_list)):
        reference_contrast_value = reference_contrast_list[reference_contrast_index]
        test_contrast_list = plot_json_data[backbone_name][f'ref_contrast_{reference_contrast_value}_test_contrast_list']
        human_gt_test_contrast_list = human_result_data[f'ref_contrast_index_{reference_contrast_index}']['y_test_contrast_average']
        if not legend_OK:
            plt.plot(rho_test_list, test_contrast_list, color=colors[reference_contrast_index], linestyle='-',
                     linewidth=2, label='Model Prediction')
            plt.plot(rho_gt_list, human_gt_test_contrast_list, color=colors[reference_contrast_index], linestyle='--',
                     linewidth=2, marker='o', label='Human Results')
            legend_OK = 1
        else:
            plt.plot(rho_test_list, test_contrast_list, color=colors[reference_contrast_index], linestyle='-', linewidth=2)
            plt.plot(rho_gt_list, human_gt_test_contrast_list, color=colors[reference_contrast_index], linestyle='--', linewidth=2, marker='o')
    plt.legend()
    plt.legend(loc='lower left')
    plt.xlabel('Test Spatial Frequency (cpd)', fontsize=12)
    plt.ylabel('Test Contrast', fontsize=12)
    plt.xlim([0.25, 25])
    plt.ylim([0.001, 1])
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(rho_x_ticks, rho_x_ticks)
    plt.yticks(y_contrast_test_ticks, y_contrast_test_ticks)
    # plt.show()
    plt.savefig(os.path.join(real_save_path, 'cosine_sim_contrast_matching'), dpi=300, bbox_inches='tight',
                pad_inches=0.02)
    plt.close()


