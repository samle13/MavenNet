import os
import h5py
import numpy as np
import pywt
from scipy import signal
import mne
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# 小波母函数

wavelet_names = ['morl']

scales_num = 9
def cwt_features(data):
    sampling_rate = 500
    scales = np.arange(1, scales_num)  # 频率范围
    coefficients_list = []

    for wavelet_name in wavelet_names:
        wavelet = pywt.ContinuousWavelet(wavelet_name)
        coefficients, _ = pywt.cwt(data, scales, wavelet, 1 / sampling_rate)
        coefficients_list.append(coefficients)

    coefficients_tensor = np.stack(coefficients_list, axis=-1)  # Stack along the last axis to get (19, 500, 127, 5)
    return coefficients_tensor




def process_and_save_data(data, output_dir, classes, patient_id,i):
    b, a = signal.butter(2, [0.002, 0.28], 'bandpass')

    for j in range(data.shape[1] // 500):
        x_clip = data[:, j*400 :(j+1) * 400+100]
        filtered_segment = np.apply_along_axis(lambda channel: signal.filtfilt(b, a, channel), axis=1, arr=x_clip)
        cwt_coeffs = cwt_features(filtered_segment)


    # Save the processed data to the HDF5 file with class name in the filename

        h5file_path = os.path.join(output_dir, f'{patient_id}_{classes}_{i}_{j}.h5')

        if not os.path.exists(h5file_path):  # Check if the file already exists
            with h5py.File(h5file_path, 'w') as h5file:
                h5file.create_dataset('cwt_coeffs', data=cwt_coeffs)


def clip_and_filter_and_save(file_dir, output_dir):
    fnames = []
    for (dirpath, dirnames, filenames) in os.walk(file_dir):
        fnames.extend(filenames)
    for filename in fnames:
        raw = mne.io.read_raw_edf(os.path.join(file_dir, filename), preload=True,encoding='latin1')
        sfreq = raw.info['sfreq']
        events_from_annot, event_dict = mne.events_from_annotations(raw)
        tmp_end = -1
        tmp_start = -1
        start = []
        end = []
        event_class = []
        for event in event_dict:
            if 'off' in event:
                tmp_end = event_dict[event]
            if 'on' in event:
                tmp_start = event_dict[event]
            if tmp_end != -1 and tmp_start != -1:
                start.append(tmp_start)
                end.append(tmp_end)
                event_class.append(event.split('on')[0])
                tmp_end = -1
                tmp_start = -1

        for i in range(len(start)):
            classes = event_class[i]
            start_time = []
            end_time = []
            for events in events_from_annot:
                if events[2] == start[i]:
                    start_time.append(events[0] / sfreq)
                if events[2] == end[i]:
                    end_time.append(events[0] / sfreq)
            # print(len(start_time))
            # print(len(end_time))
            for i in range(len(start_time)):
                raw_cropped = raw.copy()
                raw_cropped.crop(tmin=start_time[i], tmax=end_time[i])
                processed_data = raw_cropped.get_data()[0:19, :]

                patient_id = filename.split('.edf')[0]
                # Save the processed data with class name in the filename
                output_subdir = os.path.join(output_dir, classes)
                os.makedirs(output_subdir, exist_ok=True)  # 递归创建目录
                process_and_save_data(processed_data, output_subdir, classes, patient_id,i)

if __name__ == "__main__":
    file_dir = './医院标注数据2022.6.15'
    output_dir = f'./data/S_{scales_num-1}/overlap'  # 保存处理后的数据的目录
    clip_and_filter_and_save(file_dir, output_dir)