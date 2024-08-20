import pywt
import numpy as np
import os
import h5py





def cwt_features(data):
    sampling_rate = 173.61
    totalscal = 65
    wcf = pywt.central_frequency(wavelet='morl')
    # 计算对应频率的小波尺度
    cparam = 2 * wcf * totalscal
    scales = cparam/np.arange(totalscal, 1, -1)
    wavelet = pywt.ContinuousWavelet('morl')
    coefficients_list = []


    wavelet = pywt.ContinuousWavelet('morl')
    coefficients, _ = pywt.cwt(data, scales, wavelet, 1 / sampling_rate)
    coefficients_list.append(coefficients)


    return coefficients_list


def process_and_save_data(data, output_dir):

    for j in range(data.shape[0] // 500):
        # x_clip = data[j *1000 :j * 1000 + 1000] # non-overlap
        x_clip = data[j *400 :(j+1) * 400 + 100]

        cwt_coeffs = cwt_features(x_clip)
        # cwt_coeffs = x_clip
        # Save the processed data to the HDF5 file
        patient_id = os.path.basename(output_dir)
        seizure_type = patient_id[0]  # 假设类别信息在文件夹的第一个字符中
        seizure_type_mapping = {'Z': 0, 'O': 0, 'N': 1, 'F': 1, 'S': 2}
        label = seizure_type_mapping.get(seizure_type, -1)  # get从字典中获取值

        if label >= 0:
            # Save the processed data to the HDF5 file
            h5file_path = os.path.join(output_dir, f'{patient_id}_{j}.h5')

            if not os.path.exists(h5file_path):  # Check if the file already exists
                with h5py.File(h5file_path, 'w') as h5file:
                    h5file.create_dataset('cwt_coeffs', data=cwt_coeffs)
                    # h5file.create_dataset('cwt_coeffs', data=x_clip)



def clip_and_filter_and_save(data_dir, output_dir):
    data_folders = os.listdir(data_dir)

    for folder_name in data_folders:
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            data_files = os.listdir(folder_path)
            for data_file in data_files:
                data_path = os.path.join(folder_path, data_file)
                data = np.loadtxt(data_path)  # 假设txt文件中存储的是EEG数据
                patient_id = os.path.splitext(data_file)[0]  # 去除文件扩展名

                output_sub_dir = os.path.join(output_dir, patient_id)
                os.makedirs(output_sub_dir, exist_ok=True)# 在指定的路径中创建目录

                process_and_save_data(data, output_sub_dir)

if __name__ == "__main__":
    file_dir = './raw_data/'
    output_dir = './data/overlap/S_64/ABvsCDvsE'  # 保存处理后的数据的目录
    clip_and_filter_and_save(file_dir, output_dir)
