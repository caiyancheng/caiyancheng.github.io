import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

dataset_root_path = r'E:\All_Conference_Papers\CVPR25\Contrast_Masking\Noise Dataset'
ob_list = ['AB', 'DK', 'KG', 'LS']

# fig, axs = plt.subplots(2,2, figsize=(8,6))
x_contrast_mask_list_all = []
y_contrast_test_list_all = []
for observer_index in range(len(ob_list)):
    observer = ob_list[observer_index]
    csv_file_path = os.path.join(dataset_root_path, f'{observer}.csv')
    df = pd.read_csv(csv_file_path, header=None)
    data_lists = df.to_dict(orient='list')
    x_contrast_mask_list = data_lists[0]
    y_contrast_test_list = data_lists[1]
    x_contrast_mask_list_all.append(x_contrast_mask_list)
    y_contrast_test_list_all.append(y_contrast_test_list)
    # row, col = np.unravel_index(observer_index, (2, 2))
    # axs[row, col].plot(x_contrast_mask_list, y_contrast_test_list)
    # axs[row, col].set_xlim([0.01, 1])
    # axs[row, col].set_ylim([0.01, 1])
    # axs[row, col].set_xscale('log')
    # axs[row, col].set_yscale('log')
x_contrast_mask_array = np.array(x_contrast_mask_list_all)
y_contrast_test_array = np.array(y_contrast_test_list_all)
x_contrast_mask_average = np.mean(x_contrast_mask_array, axis=0)
y_contrast_test_average = np.mean(y_contrast_test_array, axis=0)
json_dict = {}
json_dict['mask_contrast_list'] = x_contrast_mask_average.tolist()
json_dict['test_contrast_list'] = y_contrast_test_average.tolist()
with open('contrast_masking_data_gabor_on_noise.json', 'w') as fp:
    json.dump(json_dict, fp)

# plt.plot(x_contrast_mask_average, y_contrast_test_average)
# plt.xlim([0.001,1])
# plt.ylim([0.01,1])
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
# plt.show()


