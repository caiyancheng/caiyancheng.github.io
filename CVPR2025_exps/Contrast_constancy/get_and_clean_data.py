import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

dataset_root_path = r'E:\All_Conference_Papers\CVPR25\Contrast_Constancy'
ob_list = ['MAG', 'GDS']
json_data_dict = {}

reference_contrast_array = np.zeros([len(ob_list), 8])
for observer_index in range(len(ob_list)):
    observer_name = ob_list[observer_index]
    df = pd.read_csv(os.path.join(dataset_root_path, f'{observer_name}_Ref_Contrast.csv'), header=None)
    data_lists = df.to_dict(orient='list')
    reference_contrast_array[observer_index,:] = data_lists[1]
average_reference_contrast = np.mean(reference_contrast_array, axis=0)
json_data_dict['average_reference_contrast'] = average_reference_contrast.tolist()

df_SpF_025_2 = pd.read_csv(os.path.join(dataset_root_path, f'{observer_name}_SpF_025_2.csv'), header=None)
df_SpF_5_25 = pd.read_csv(os.path.join(dataset_root_path, f'{observer_name}_SpF_5_25.csv'), header=None)
data_lists_SpF_025_2 = df_SpF_025_2.to_dict(orient='list')
data_lists_SpF_5_25 = df_SpF_5_25.to_dict(orient='list')
plt.figure(figsize=(10,8))
for ref_contrast_index in range(len(average_reference_contrast)):
    ref_contrast_value = average_reference_contrast[ref_contrast_index]
    x_test_frequency = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 25]
    y_test_contrast_array = np.zeros([len(ob_list), 9])
    for observer_index in range(len(ob_list)):
        observer_name = ob_list[observer_index]
        y_test_contrast_array[observer_index, 0:4] = data_lists_SpF_025_2[1][4 * ref_contrast_index:4 * ref_contrast_index + 4]
        y_test_contrast_array[observer_index, 4:9] = data_lists_SpF_5_25[1][5 * ref_contrast_index:5 * ref_contrast_index + 5]
    y_test_contrast_average = np.mean(y_test_contrast_array, axis=0)
    plt.plot(x_test_frequency, y_test_contrast_average)
    json_data_dict[f'ref_contrast_index_{ref_contrast_index}'] = {'x_test_frequency': x_test_frequency,
                                                                  'y_test_contrast_average': y_test_contrast_average.tolist()}
plt.xlim([0.25, 25])
plt.ylim([0.003, 1])
plt.xscale('log')
plt.yscale('log')
plt.show()

with open('contrast_constancy_sin_5_cpd.json', 'w') as fp:
    json.dump(json_data_dict, fp)


