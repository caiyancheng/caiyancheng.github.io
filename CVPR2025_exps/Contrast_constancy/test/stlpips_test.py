import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Sinusoidal_grating_generator.generate_plot_sinusoidal_grating import generate_sinusoidal_grating
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
from stlpips_pytorch import stlpips
import os
from display_encoding import display_encode
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar
import scipy.optimize as opt
display_encode_tool = display_encode(400)

# loss_fn_alex = lpips.LPIPS(net='alex').eval()
# loss_fn_vgg = lpips.LPIPS(net='vgg').eval()

torch.hub.set_dir(r'E:\Torch_hub')
save_root_path = 'new_data_logs/stlpips/different_rho'
os.makedirs(save_root_path, exist_ok=True)
all_lpips_model_list = ['alex', 'vgg']

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
json_plot_data['backbone_name_list'] = all_lpips_model_list
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
R_vid_Con_0_ct = torch.tensor(display_encode_tool.L2C_sRGB(np.stack([R_vid_Con_0] * 3, axis=-1)), dtype=torch.float32).permute(2, 0, 1)[None, ...]
R_vid_Con_1_ct = torch.tensor(display_encode_tool.L2C_sRGB(np.stack([R_vid_Con_1] * 3, axis=-1)), dtype=torch.float32).permute(2, 0, 1)[None, ...]
norm_R_vid_Con_0_ct = (R_vid_Con_0_ct - 0.5) * 2
norm_R_vid_Con_1_ct = (R_vid_Con_1_ct - 0.5) * 2
def test_feature_lpips_loss(contrast, rho, lpips_loss):
    T_vid = generate_sinusoidal_grating(W=default_W, H=default_H,
                                        spatial_frequency=rho,
                                        orientation=default_O, L_b=default_L_b,
                                        contrast=contrast, ppd=default_ppd)
    T_vid_ct = torch.tensor(display_encode_tool.L2C_sRGB(np.stack([T_vid] * 3, axis=-1)), dtype=torch.float32).permute(2, 0, 1)[None, ...]
    norm_T_vid_ct = (T_vid_ct - 0.5) * 2
    T_lpips_loss = float(lpips_loss(norm_R_vid_Con_0_ct, norm_T_vid_ct))
    return T_lpips_loss
def T_optimize_target(contrast, rho, lpips_loss, max_T_lpips_loss, aim_contrast_matching_scale):
    T_lpips_loss = test_feature_lpips_loss(contrast, rho, lpips_loss)
    contrast_matching_scale = T_lpips_loss / max_T_lpips_loss
    return contrast_matching_scale - aim_contrast_matching_scale

for backbone_name in tqdm(all_lpips_model_list):
    json_plot_data[backbone_name] = {}
    lpips_loss = stlpips.LPIPS(net=backbone_name, variant="shift_tolerant").eval()
    max_R_lpips_loss = float(lpips_loss(norm_R_vid_Con_0_ct, norm_R_vid_Con_1_ct))
    for reference_contrast_index in range(len(reference_contrast_list)):
        reference_contrast_value = reference_contrast_list[reference_contrast_index]
        R_vid = generate_sinusoidal_grating(W=default_W, H=default_H,
                                            spatial_frequency=default_rho_referenece,
                                            orientation=default_O, L_b=default_L_b,
                                            contrast=reference_contrast_value, ppd=default_ppd)
        R_vid_ct = torch.tensor(display_encode_tool.L2C_sRGB(np.stack([R_vid] * 3, axis=-1)), dtype=torch.float32).permute(2, 0, 1)[None, ...]
        norm_R_vid_ct = (R_vid_ct - 0.5) * 2
        R_lpips_loss = float(lpips_loss(norm_R_vid_Con_0_ct, norm_R_vid_ct))
        contrast_matching_scale = R_lpips_loss / max_R_lpips_loss
        contrast_matching_scale = max(min(contrast_matching_scale, 1), 0)
        match_test_contrast_list = []
        for rho_test_index in range(len(rho_test_list)):
            rho_test_value = rho_test_list[rho_test_index]
            T_vid_Con_1 = generate_sinusoidal_grating(W=default_W, H=default_H,
                                                      spatial_frequency=rho_test_value,
                                                      orientation=default_O, L_b=default_L_b,
                                                      contrast=1, ppd=default_ppd)
            T_vid_Con_1_ct = torch.tensor(display_encode_tool.L2C_sRGB(np.stack([T_vid_Con_1] * 3, axis=-1)),
                                          dtype=torch.float32).permute(2, 0, 1)[None, ...]
            norm_T_vid_Con_1_ct = (T_vid_Con_1_ct - 0.5) * 2
            max_T_lpips_loss = float(lpips_loss(norm_R_vid_Con_0_ct, norm_T_vid_Con_1_ct))
            # initial_test_contrast = json_data[f'ref_contrast_index_{reference_contrast_index}']['y_test_contrast_average'][rho_test_index]
            bounds = [0.001,1]
            target_function = lambda test_contrast: T_optimize_target(
                contrast=test_contrast, rho=rho_test_value, lpips_loss=lpips_loss,
                max_T_lpips_loss=max_T_lpips_loss, aim_contrast_matching_scale=contrast_matching_scale
            )
            # test_contrast_optimization_result = minimize_scalar(target_function, bounds=bounds, method='bounded', tol=1e-5)
            # try:
            #     test_contrast_optimization_result = root_scalar(target_function, bracket=bounds, xtol=1e-5)
            #     test_contrast_optimization_value = test_contrast_optimization_result.root
            # except:
            #     if target_function(1) < 0:
            #         test_contrast_optimization_value = 1
            #     else:
            #         test_contrast_optimization_value = 0.001
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
            df.to_csv(os.path.join(save_root_path, f'stlpips_test_ppd_{default_ppd}_temporary.csv'), index=False)
        json_plot_data[backbone_name][f'ref_contrast_{reference_contrast_value}_test_contrast_list'] = match_test_contrast_list

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'stlpips_test_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'stlpips_test_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)
