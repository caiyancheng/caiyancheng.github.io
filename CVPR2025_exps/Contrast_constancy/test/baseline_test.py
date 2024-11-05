import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Sinusoidal_grating_generator.generate_plot_sinusoidal_grating import generate_sinusoidal_grating
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import os
from display_encoding import display_encode
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar
import scipy.optimize as opt
display_encode_tool = display_encode(400)

torch.hub.set_dir(r'E:\Torch_hub')
save_root_path = 'new_data_logs/baseline/different_rho'
os.makedirs(save_root_path, exist_ok=True)
all_backbone_list = ['baseline']

default_W = 224
default_H = 224
default_R = 1
default_rho_referenece = 5

rho_test_list_gt = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]
rho_test_list_2 = np.logspace(np.log10(0.25), np.log10(25), 20).tolist()
rho_test_list_gt = [round(x, 3) for x in rho_test_list_gt]
rho_test_list_2 = [round(x, 3) for x in rho_test_list_2]
rho_test_list = sorted(set(rho_test_list_gt + rho_test_list_2))


with open(r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Contrast_constancy/contrast_constancy_sin_5_cpd.json', 'r') as fp:
    json_data = json.load(fp)
reference_contrast_list = json_data['average_reference_contrast']
# contrast_test_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_O = 0
default_L_b = 10
default_ppd = 60

csv_data = {}
csv_data['backbone_name'] = []
csv_data['rho_test'] = []
csv_data['test_contrast'] = []

json_plot_data = {}
json_plot_data['backbone_name_list'] = all_backbone_list
json_plot_data['reference_contrast_list'] = reference_contrast_list
json_plot_data['rho_test_list'] = rho_test_list

R_vid_Con_0 = generate_sinusoidal_grating(W=default_W, H=default_H,
                                              spatial_frequency=default_rho_referenece,
                                              orientation=default_O, L_b=default_L_b,
                                              contrast=0, ppd=default_ppd)
R_vid_Con_1 = generate_sinusoidal_grating(W=default_W, H=default_H,
                                          spatial_frequency=default_rho_referenece,
                                          orientation=default_O, L_b=default_L_b,
                                          contrast=1, ppd=default_ppd)
R_vid_Con_0_ct = torch.tensor(display_encode_tool.L2C_sRGB(np.stack([R_vid_Con_0] * 3, axis=-1)), dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
R_vid_Con_1_ct = torch.tensor(display_encode_tool.L2C_sRGB(np.stack([R_vid_Con_1] * 3, axis=-1)), dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()


def test_feature_cos_similarity(contrast, rho):
    T_vid = generate_sinusoidal_grating(W=default_W, H=default_H,
                                        spatial_frequency=rho,
                                        orientation=default_O, L_b=default_L_b,
                                        contrast=contrast, ppd=default_ppd)
    T_vid = np.stack([T_vid] * 3, axis=-1)
    T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
    T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
    T_vid_ct_feature = T_vid_ct
    T_cos_similarity = float(F.cosine_similarity(R_vid_Con_0_ct_feature.reshape(1, -1), T_vid_ct_feature.reshape(1, -1)).cpu())
    return T_cos_similarity
def T_optimize_target(contrast, rho, min_T_cos_similarity, aim_contrast_matching_scale):
    T_cos_similarity = test_feature_cos_similarity(contrast, rho)
    contrast_matching_scale = (1 - T_cos_similarity) / (1 - min_T_cos_similarity)
    return contrast_matching_scale - aim_contrast_matching_scale

for backbone_name in tqdm(all_backbone_list):
    json_plot_data[backbone_name] = {}
    R_vid_Con_0_ct_feature = R_vid_Con_0_ct
    R_vid_Con_1_ct_feature = R_vid_Con_1_ct
    min_R_cos_similarity = float(F.cosine_similarity(R_vid_Con_0_ct_feature.reshape(1, -1), R_vid_Con_1_ct_feature.reshape(1, -1)).cpu())
    for reference_contrast_index in range(len(reference_contrast_list)):
        reference_contrast_value = reference_contrast_list[reference_contrast_index]
        R_vid = generate_sinusoidal_grating(W=default_W, H=default_H,
                                            spatial_frequency=default_rho_referenece,
                                            orientation=default_O, L_b=default_L_b,
                                            contrast=reference_contrast_value, ppd=default_ppd)
        R_vid = np.stack([R_vid] * 3, axis=-1)
        R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
        R_vid_ct_feature = R_vid_ct
        R_cos_similarity = float(F.cosine_similarity(R_vid_Con_0_ct_feature.reshape(1, -1), R_vid_ct_feature.reshape(1, -1)).cpu())
        contrast_matching_scale = (1 - R_cos_similarity) / (1 - min_R_cos_similarity)
        contrast_matching_scale = max(min(contrast_matching_scale, 1), 0)
        match_test_contrast_list = []
        for rho_test_index in range(len(rho_test_list)):
            rho_test_value = rho_test_list[rho_test_index]
            T_vid_Con_1 = generate_sinusoidal_grating(W=default_W, H=default_H,
                                                      spatial_frequency=rho_test_value,
                                                      orientation=default_O, L_b=default_L_b,
                                                      contrast=1, ppd=default_ppd)
            T_vid_Con_1_ct = torch.tensor(display_encode_tool.L2C_sRGB(np.stack([T_vid_Con_1] * 3, axis=-1)),
                                          dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
            T_vid_Con_1_ct_feature = T_vid_Con_1_ct
            min_T_cos_similarity = float(F.cosine_similarity(R_vid_Con_0_ct_feature.reshape(1, -1), T_vid_Con_1_ct_feature.reshape(1, -1)).cpu())
            # initial_test_contrast = json_data[f'ref_contrast_index_{reference_contrast_index}']['y_test_contrast_average'][rho_test_index]
            bounds = [0.001,1]
            target_function = lambda test_contrast: T_optimize_target(
                contrast=test_contrast, rho=rho_test_value,
                min_T_cos_similarity=min_T_cos_similarity, aim_contrast_matching_scale=contrast_matching_scale
            )
            # test_contrast_optimization_result = minimize_scalar(target_function, bounds=bounds, method='bounded', tol=1e-5)
            if target_function(bounds[0]) > 0:
                test_contrast_optimization_value = 0.001
            else:
                test_contrast_optimization_result = root_scalar(target_function, bracket=bounds, xtol=1e-5)
                test_contrast_optimization_value = test_contrast_optimization_result.root

            csv_data['backbone_name'].append(backbone_name)
            csv_data['rho_test'].append(rho_test_value)
            csv_data['test_contrast'].append(test_contrast_optimization_value)
            match_test_contrast_list.append(test_contrast_optimization_value)
            df = pd.DataFrame(csv_data)
            df.to_csv(os.path.join(save_root_path, f'baseline_test_ppd_{default_ppd}_temporary.csv'), index=False)
        json_plot_data[backbone_name][f'ref_contrast_{reference_contrast_value}_test_contrast_list'] = match_test_contrast_list

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'baseline_test_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'baseline_test_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)
