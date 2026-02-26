import os
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

def process_cc200_strict_slice(excel_path, mat_dir, output_path):
    print(f"Reading: {excel_path}")

    try:
        df_mdd = pd.read_excel(excel_path, sheet_name='MDD')
        df_controls = pd.read_excel(excel_path, sheet_name='Controls')
    except Exception as e:
        print(f"Excel read error: {e}")
        return

    cc200_slice = slice(228, 428)
    min_required_cols = 428

    data_dict = {
        'id': [],
        'label': [],
        'cc200_bold': [],
        'cc200_pcc': []
    }

    seen_time_points = set()

    def process_sample(row, is_mdd):
        sub_id = str(row['ID']).strip()
        mat_filename = f"ROISignals_{sub_id}.mat"
        mat_file_path = os.path.join(mat_dir, mat_filename)

        if not os.path.exists(mat_file_path):
            return False

        if is_mdd:
            label = np.array([0, 1], dtype=np.float32)
        else:
            label = np.array([1, 0], dtype=np.float32)

        try:
            mat_data = sio.loadmat(mat_file_path)
            signal_key = None
            for k in mat_data.keys():
                if not k.startswith('__'):
                    signal_key = k
                    break

            if signal_key is None:
                return False

            full_signal = mat_data[signal_key]
            time_points, n_rois = full_signal.shape

            if time_points not in seen_time_points:
                group_name = "MDD" if is_mdd else "Ctrl"
                print(f"\nNew shape: {time_points}x{n_rois} ({group_name}, {sub_id})")
                seen_time_points.add(time_points)

            if n_rois >= min_required_cols:
                roi_sig = full_signal[:, cc200_slice]
            else:
                return False

            if roi_sig.shape[1] != 200:
                print(f"Shape error: {sub_id}, {roi_sig.shape}")
                return False

            roi_sig_t = roi_sig.T
            pcc = np.corrcoef(roi_sig_t)
            pcc = np.nan_to_num(pcc)

            data_dict['id'].append(sub_id)
            data_dict['label'].append(label)
            data_dict['cc200_bold'].append(roi_sig_t)
            data_dict['cc200_pcc'].append(pcc)

            return True

        except Exception as e:
            print(f"Error ({sub_id}): {e}")
            return False

    print("Processing MDD...")
    count_mdd = 0
    for idx, row in tqdm(df_mdd.iterrows(), total=df_mdd.shape[0], desc="MDD"):
        if process_sample(row, is_mdd=True):
            count_mdd += 1

    print("Processing Controls...")
    count_ctrl = 0
    for idx, row in tqdm(df_controls.iterrows(), total=df_controls.shape[0], desc="Ctrl"):
        if process_sample(row, is_mdd=False):
            count_ctrl += 1

    print(f"\nDone. MDD: {count_mdd}, Ctrl: {count_ctrl}")
    print(f"Timepoints: {sorted(list(seen_time_points))}")

    final_save_dict = {}
    final_save_dict['id'] = np.array(data_dict['id'])
    final_save_dict['label'] = np.array(data_dict['label'], dtype=np.float32)
    final_save_dict['cc200_pcc'] = np.array(data_dict['cc200_pcc'], dtype=np.float32)

    if len(seen_time_points) > 1:
        print("Variable timepoints, saving as object array.")
        n_samples = len(data_dict['cc200_bold'])
        obj_arr = np.empty(n_samples, dtype=object)
        for i, arr in enumerate(data_dict['cc200_bold']):
            obj_arr[i] = arr
        final_save_dict['cc200_bold'] = obj_arr
    else:
        print("Consistent timepoints, saving as numpy array.")
        final_save_dict['cc200_bold'] = np.array(data_dict['cc200_bold'], dtype=np.float32)

    print(f"Saving to {output_path}...")
    np.savez(output_path, **final_save_dict)
    print("Saved.")

mat_folder = r"/home/path/dataset/rest-meta-mdd/ROISignals_FunImgARCWF/"
excel_file = r"/home/path/dataset/rest-meta-mdd/REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx"
output_file = "./rest-meta-mdd_cc200.npz"

if __name__ == "__main__":
    if not os.path.exists(mat_folder):
        print(f"Dir not found: {mat_folder}")
    elif not os.path.exists(excel_file):
        print(f"File not found: {excel_file}")
    else:
        process_cc200_strict_slice(excel_file, mat_folder, output_file)