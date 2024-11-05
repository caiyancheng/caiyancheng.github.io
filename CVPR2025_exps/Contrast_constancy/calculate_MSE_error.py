import json
import os
import numpy as np
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr

result_root_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Contrast_constancy\test\new_data_logs'
tested_models = ['baseline', 'cvvdp', 'dino', 'dinov2', 'lpips', 'mae', 'openclip', 'sam', 'sam2', 'stlpips', 'vae']
# tested_models = ['openclip']
rho_gt_list = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]
gt_rho_index = [0, 3, 7, 11, 16, 20, 22, 25, 26]

human_result_json_path = r'E:\Py_codes\LVM_Comparision\Feature_Similarity_paper_report\Contrast_constancy/contrast_constancy_sin_5_cpd.json'
with open(human_result_json_path, 'r') as fp:
    human_result_data = json.load(fp)

reference_contrast_list = human_result_data['average_reference_contrast']
human_result_array = np.zeros([len(reference_contrast_list), len(rho_gt_list)])
for reference_contrast_index, reference_contrast_value in enumerate(reference_contrast_list):
    human_result_array[reference_contrast_index, :] = human_result_data[f'ref_contrast_index_{reference_contrast_index}']['y_test_contrast_average']

for model_index, model_name in enumerate(tested_models):
    model_path = os.path.join(result_root_path, model_name ,'different_rho', f'{model_name}_test_ppd_60_final.json')
    with open(model_path, 'r') as fp:
        test_result = json.load(fp)
    model_prediction_result_array = np.zeros([len(reference_contrast_list), len(rho_gt_list)])
    if model_name == 'openclip':
        clip_model_name_list = test_result['clip_model_name']
        clip_model_trainset_list = test_result['clip_model_trainset']
        for clip_model_name_index, clip_model_name_value in enumerate(clip_model_name_list):
            clip_model_trainset_value = clip_model_trainset_list[clip_model_name_index]
            for reference_contrast_index, reference_contrast_value in enumerate(reference_contrast_list):
                test_result_full_list = test_result[f'{clip_model_name_value}-{clip_model_trainset_value}'][
                    f'ref_contrast_{reference_contrast_value}_test_contrast_list']
                model_prediction_result_array[reference_contrast_index, :] = [test_result_full_list[i] for i in
                                                                              gt_rho_index]
            log_RMSE_loss = root_mean_squared_error(np.log10(human_result_array), np.log10(model_prediction_result_array))
            # log_Pearson_list = []
            # for index in range(human_result_array.shape[0]):
            #     log_Pearson_list.append(pearsonr(np.log10(human_result_array[index,:]), np.log10(model_prediction_result_array[index,:]))[0])
            # log_Pearson = np.nanmean(np.array(log_Pearson_list))
            print('Backbone: ', f'{model_name}: {clip_model_name_value}-{clip_model_trainset_value}')
            print('RMSE: ', round(log_RMSE_loss,4))
            # print('Pearson: ', round(log_Pearson, 4))
    else:
        backbone_name_list = test_result['backbone_name_list']
        for backbone_name_index, backbone_name in enumerate(backbone_name_list):
            for reference_contrast_index, reference_contrast_value in enumerate(reference_contrast_list):
                model_reference_contrast_list = test_result['reference_contrast_list']
                test_result_full_list = test_result[backbone_name][
                    f'ref_contrast_{reference_contrast_value}_test_contrast_list']
                model_prediction_result_array[reference_contrast_index, :] = [test_result_full_list[i] for i in
                                                                              gt_rho_index]
            log_RMSE_loss = root_mean_squared_error(np.log10(human_result_array), np.log10(model_prediction_result_array))
            # log_Pearson_list = []
            # for index in range(human_result_array.shape[0]):
            #     log_Pearson_list.append(
            #         pearsonr(np.log10(human_result_array[index, :]), np.log10(model_prediction_result_array[index, :]))[0])
            # log_Pearson = np.nanmean(np.array(log_Pearson_list))
            print('Backbone: ', f'{model_name}: {backbone_name}')
            print('RMSE: ', round(log_RMSE_loss,4))
            # print('Pearson: ', round(log_Pearson, 4))



