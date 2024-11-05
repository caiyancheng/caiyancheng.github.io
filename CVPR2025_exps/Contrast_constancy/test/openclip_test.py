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
import open_clip
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar
import scipy.optimize as opt
from PIL import Image
display_encode_tool = display_encode(400)

torch.hub.set_dir(r'E:\Torch_hub')
save_root_path = 'new_data_logs/openclip/different_rho'
os.makedirs(save_root_path, exist_ok=True)
all_clip_model_list = [('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN101', 'openai'), ('RN101', 'yfcc15m'),
                       ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k'), ('ViT-B-16', 'openai'), ('ViT-B-16', 'laion2b_s34b_b88k'),
                       ('ViT-L-14', 'openai'), ('ViT-L-14', 'laion2b_s32b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
                       ('convnext_large_d', 'laion2b_s26b_b102k_augreg'), ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg')]

default_W = 224
default_H = 224
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
csv_data['clip_model_name'] = []
csv_data['clip_model_trainset'] = []
csv_data['rho_test'] = []
csv_data['test_contrast'] = []

json_plot_data = {}
json_plot_data['clip_model_name'] = []
json_plot_data['clip_model_trainset'] = []
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
R_vid_Con_0_ct = Image.fromarray((display_encode_tool.L2C_sRGB(np.stack([R_vid_Con_0] * 3, axis=-1)) * 255).astype(np.uint8))
R_vid_Con_1_ct = Image.fromarray((display_encode_tool.L2C_sRGB(np.stack([R_vid_Con_1] * 3, axis=-1)) * 255).astype(np.uint8))

def test_feature_cos_similarity(contrast, rho, model, preprocess):
    T_vid = generate_sinusoidal_grating(W=default_W, H=default_H,
                                        spatial_frequency=rho,
                                        orientation=default_O, L_b=default_L_b,
                                        contrast=contrast, ppd=default_ppd)
    T_vid_ct = Image.fromarray((display_encode_tool.L2C_sRGB(np.stack([T_vid] * 3, axis=-1)) * 255).astype(np.uint8))
    T_vid_ct_feature = model.encode_image(preprocess(T_vid_ct).unsqueeze(0).cuda())
    T_cos_similarity = float(F.cosine_similarity(R_vid_Con_0_ct_feature, T_vid_ct_feature).cpu())
    return T_cos_similarity
def T_optimize_target(contrast, rho, model, preprocess, min_T_cos_similarity, aim_contrast_matching_scale):
    T_cos_similarity = test_feature_cos_similarity(contrast, rho, model, preprocess)
    contrast_matching_scale = (1 - T_cos_similarity) / (1 - min_T_cos_similarity)
    return contrast_matching_scale - aim_contrast_matching_scale

for clip_model_cell in tqdm(all_clip_model_list):
    clip_model_name = clip_model_cell[0]
    clip_model_trainset = clip_model_cell[1]
    json_plot_data['clip_model_name'].append(clip_model_name)
    json_plot_data['clip_model_trainset'].append(clip_model_trainset)
    json_plot_data[f'{clip_model_name}-{clip_model_trainset}'] = {}
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_model_trainset,
                                                                 cache_dir=r'E:\Openclip_cache')
    model.eval()
    model.cuda()

    R_vid_Con_0_ct_feature = model.encode_image(preprocess(R_vid_Con_0_ct).unsqueeze(0).cuda())
    R_vid_Con_1_ct_feature = model.encode_image(preprocess(R_vid_Con_1_ct).unsqueeze(0).cuda())
    min_R_cos_similarity = float(F.cosine_similarity(R_vid_Con_0_ct_feature, R_vid_Con_1_ct_feature).cpu())
    for reference_contrast_index in range(len(reference_contrast_list)):
        reference_contrast_value = reference_contrast_list[reference_contrast_index]
        R_vid = generate_sinusoidal_grating(W=default_W, H=default_H,
                                            spatial_frequency=default_rho_referenece,
                                            orientation=default_O, L_b=default_L_b,
                                            contrast=reference_contrast_value, ppd=default_ppd)
        R_vid_ct = Image.fromarray((display_encode_tool.L2C_sRGB(np.stack([R_vid] * 3, axis=-1)) * 255).astype(np.uint8))
        R_vid_ct_feature = model.encode_image(preprocess(R_vid_ct).unsqueeze(0).cuda())
        R_cos_similarity = float(F.cosine_similarity(R_vid_Con_0_ct_feature, R_vid_ct_feature).cpu())
        contrast_matching_scale = (1 - R_cos_similarity) / (1 - min_R_cos_similarity)
        contrast_matching_scale = max(min(contrast_matching_scale, 1), 0)
        match_test_contrast_list = []
        for rho_test_index in range(len(rho_test_list)):
            rho_test_value = rho_test_list[rho_test_index]
            T_vid_Con_1 = generate_sinusoidal_grating(W=default_W, H=default_H,
                                                      spatial_frequency=rho_test_value,
                                                      orientation=default_O, L_b=default_L_b,
                                                      contrast=1, ppd=default_ppd)
            T_vid_Con_1_ct = Image.fromarray((display_encode_tool.L2C_sRGB(np.stack([T_vid_Con_1] * 3, axis=-1)) * 255).astype(np.uint8))
            T_vid_Con_1_ct_feature = model.encode_image(preprocess(T_vid_Con_1_ct).unsqueeze(0).cuda())
            min_T_cos_similarity = float(F.cosine_similarity(R_vid_Con_0_ct_feature, T_vid_Con_1_ct_feature).cpu())
            # initial_test_contrast = json_data[f'ref_contrast_index_{reference_contrast_index}']['y_test_contrast_average'][rho_test_index]
            bounds = [0.001,1]
            target_function = lambda test_contrast: T_optimize_target(
                contrast=test_contrast, rho=rho_test_value, model=model, preprocess=preprocess,
                min_T_cos_similarity=min_T_cos_similarity, aim_contrast_matching_scale=contrast_matching_scale
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
            csv_data['clip_model_name'].append(clip_model_name)
            csv_data['clip_model_trainset'].append(clip_model_trainset)
            csv_data['rho_test'].append(rho_test_value)
            csv_data['test_contrast'].append(test_contrast_optimization_value)
            match_test_contrast_list.append(test_contrast_optimization_value)
            df = pd.DataFrame(csv_data)
            df.to_csv(os.path.join(save_root_path, f'openclip_test_ppd_{default_ppd}_temporary.csv'), index=False)
        json_plot_data[f'{clip_model_name}-{clip_model_trainset}'][f'ref_contrast_{reference_contrast_value}_test_contrast_list'] = match_test_contrast_list

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'openclip_test_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'openclip_test_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)
