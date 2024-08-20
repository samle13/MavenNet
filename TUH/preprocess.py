import pywt
import numpy as np
import collections
import pickle as pkl
from scipy import signal
import matplotlib.pyplot as plt
import os
import h5py


seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id', 'seizure_type', 'data'])


wavelet_names = ['morl']
def cwt_features(data):
    sampling_rate = 250
    totalscal = 64
    coefficients_list = []


    # 中心频率
    wcf = pywt.central_frequency(wavelet='morl')
    # 计算对应频率的小波尺度
    cparam = 2 * wcf * totalscal
    scales = cparam/np.arange(totalscal, 1, -1)
    wavelet = pywt.ContinuousWavelet('morl')
    coefficients, _ = pywt.cwt(data, scales, wavelet, 1 / sampling_rate)
    coefficients_list.append(coefficients)

    
    return coefficients_list

patient_id_list=[]
seizure_list=[]
def process_and_save_data(data, output_dir):
    b, a = signal.butter(2, [0.0016, 0.198], 'bandpass')

    patient_id_list.append(os.path.basename(output_dir).split('_')[3])


    #print(os.path.basename(output_dir).split('_')[3])
    for j in range(data.shape[1] // 500):
        x_clip = data[:, j * 400:j * 400 + 500]
        filtered_segment = np.apply_along_axis(lambda channel: signal.filtfilt(b, a, channel), axis=1, arr=x_clip)
        cwt_coeffs = cwt_features(filtered_segment)


        patient_id = os.path.basename(output_dir)

        h5file_path = os.path.join(output_dir, f'{patient_id}_{j}.h5')

        if not os.path.exists(h5file_path):  # Check if the file already exists
            with h5py.File(h5file_path, 'w') as h5file:
                h5file.create_dataset('cwt_coeffs', data=cwt_coeffs)


def process_and_save_batch(batch, output_dir):
    for x in batch:
        process_and_save_data(x.data, output_dir)


def clip_and_filter_and_save():
    pkl_list = os.listdir('G:/kayla go to study/TUH患者分类/test')

    for pkl_file in pkl_list:
        data = pkl.load(open(os.path.join('G:/kayla go to study/TUH患者分类/test', pkl_file), 'rb'))
        patient_id = os.path.splitext(pkl_file)[0]  # Remove the file extension

        output_dir = os.path.join('G:/kayla go to study/TUH患者分类/preprocess/test', patient_id)
        os.makedirs(output_dir, exist_ok=True)

        process_and_save_data(data.data, output_dir)


clip_and_filter_and_save()

# print(np.unique(patient_id_list))