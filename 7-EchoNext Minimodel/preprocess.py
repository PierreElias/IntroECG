import pandas as pd 
import numpy as np 
from multiprocessing import Pool
import joblib 
import json

import os
from argparse import ArgumentParser

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

TABULAR_COLS = ['sex', 'age_at_ecg', 'ventricular_rate', 'atrial_rate', 
                'pr_interval', 'qrs_duration',  'qt_corrected']

TABULAR_FLOAT = ['age_at_ecg', 'ventricular_rate', 'atrial_rate', 
                'pr_interval', 'qrs_duration',  'qt_corrected']

TABULAR_CLEAN = [col + '_clean' for col in TABULAR_COLS]


def tabular_transformer(df_tab, fit_yn=False, pipe=None):
    """
    This function is used to normalize train/test dataset using standard scaler and median impute.
    Pipeline should be fit on train set and apply to train/val/test sets. 
    Note: 
        1. sex is not allowed to be missing, and needs to be male/female
        2. Atrial rate and PR interval are assumed 0 if missing, presumably caused by atrial arrhythmia
    
    """
    dftab = df_tab.copy()
    dftab['sex_clean'] = dftab['sex'].str.lower().map(lambda x: 1 if x=='male' else 0 if x=='female' else np.nan)

    dftab['atrial_rate'] = dftab['atrial_rate'].fillna(0)
    dftab['pr_interval'] = dftab['pr_interval'].fillna(0)
    
    # standard scaling and impute with median
    fea_cols = TABULAR_FLOAT
    if fit_yn: 
        # fit pipeline
        pipe = Pipeline([('scale', StandardScaler()), 
                         ('impute', SimpleImputer(strategy='median'))
                         ])
        tab_scaled = pipe.fit_transform(dftab[fea_cols])
    else:
        # itransform test/validation data
        assert pipe is not None, 'Please provide a trained normalization pipeline to transform data!'
        tab_scaled = pipe.transform(dftab[fea_cols])

    cols = dftab[fea_cols].add_suffix('_clean').columns
    dftab[cols] = tab_scaled
    
    return dftab[TABULAR_CLEAN], pipe


def extract_waveform(ecg_list, norm_file, n_jobs=10):
        
    with Pool(processes=n_jobs) as pm:
        waveforms = pm.map(np.load, ecg_list)
    
    waveforms = np.array(waveforms)

    # truncate and normalize ecg waveforms by lead
    with open(norm_file, 'r') as fid:
        params = json.load(fid)

    waveforms = per_lead_truncation_normalization(waveforms, params)

    return waveforms, params


def reshape_array(ecgdata):
    "reshape stacked ECG waveform numpy arrays to (N, 12, 2500, 1)"

    if ecgdata.shape[1:] == (2500,12,1):
        print('data shape:',ecgdata.shape,'transposing now...')
        ecgdata = np.transpose(ecgdata, axes=[0,2,1,3])
    elif ecgdata.shape[1:] == (1,2500,12):
        print('data shape:', ecgdata.shape, 'transposing now...')
        ecgdata = np.transpose(ecgdata, axes=[0,3,2,1])
    elif ecgdata.shape[1:] == (12,2500,1):
        pass
    assert ecgdata.shape[1:] == (12, 2500, 1), "dataset is not X,12,2500,1"

    return ecgdata


def per_lead_truncation_normalization(data, limits):

    data = reshape_array(data)
    data_norm = data
    mean_pre, std_pre = np.mean(data, axis=(0, 2, 3)), np.std(data, axis=(0, 2, 3))
    assert (len(limits['lowerbound']) == data_norm.shape[1]), "shape of normalization params doesn't match data.shape[1]"
    for lead in range(len(limits['lowerbound'])):
        data_norm[:, lead, :, :] = np.where(data_norm[:, lead, :, :] > limits['upperbound'][lead], limits['upperbound'][lead], data_norm[:, lead, :, :])
        data_norm[:, lead, :, :] = np.where(data_norm[:, lead, :, :] < limits['lowerbound'][lead], limits['lowerbound'][lead], data_norm[:, lead, :, :])

        print("pre-normalizing min:",data_norm[:, lead, :, :].min())
        print("pre-normalizing max:", data_norm[:, lead, :, :].max())
        data_norm[:, lead, :, :] = (data_norm[:, lead, :, :] - limits['mean'][lead]) / limits['std'][lead]
        print("post-normalizing min:",data_norm[:, lead, :, :].min())
        print("post-normalizing max:", data_norm[:, lead, :, :].max())

    print('Mean and std before normalization: \n{}\n{}'.format(mean_pre, std_pre))
    mean_norm, std_norm = np.mean(data_norm, axis=(0, 2, 3)), np.std(data_norm, axis=(0, 2, 3))
    print('Mean and STD after normalization: \n{}\n{}'.format(mean_norm, std_norm))

    # Transpose data for model
    data_norm = np.transpose(data_norm, axes=[0, 3, 2, 1])
    print('Normalized data shape for model:', data_norm.shape)
    assert data_norm.shape[1:] == (1, 2500, 12), "dataset shape is not (X, 1, 2500, 12)"

    return data_norm


if __name__ == '__main__':
    ap = ArgumentParser(description='Preprocessing ECG and tabular features for echonext pipeline')
    ap.add_argument('--ecgmeta_path', type=str, help='Path to ECG metadata parquet file')
    # ap.add_argument('--ecg_dir', type=str, help='Directory of ECG numpy files')
    ap.add_argument('--out_dir', type=str, help='Output data directory')
    ap.add_argument(
        '--tabular_pipeline_path',
        type=str,
        help='path to tabular preprocessing pipeline')
    ap.add_argument(
        '--waveform_params_path', 
        type=str,
        help='path to waveform normalization parameter file')
    ap.add_argument('--n_jobs', '-p', type=int, default=10, help='Number of parallel processes')
    args = ap.parse_args()

    # load metadata
    data = pd.read_parquet(args.ecgmeta_path)
    print(data.shape)
    # keep ECGs with preprcoessed waveforms 
    print('Missing waveforms: {}'.format((data.file_created==0).sum()))
    data = data.query("file_created==1")

    # set unknown sex to 
    data['sex'] = data['sex'].str.lower()
    data.loc[~data['sex'].isin(['male', 'female']), 'sex'] = None

    # remove pediatric, poor quality, ventricular paced ecgs, missing sex
    to_excl = (data['age_at_ecg']<18) | (data['age_at_ecg'].isnull()) | (data['poor_data_quality_flag']==1) | (data['ventricular_pacing_flag']==1) | (data['sex'].isnull())
    print('Remove {} ecgs with age<18, missing age/sex, poor quality or ventricular pacing'.format(to_excl.sum()))
    data = data[~to_excl]

    # remove ECGs where all measurements are 0/NaN
    ecgm_cols = ['ventricular_rate', 'atrial_rate', 'pr_interval', 'qrs_duration', 'qt_corrected']
    to_excl = ((data[ecgm_cols] == 0) | (data[ecgm_cols].isnull())).all(axis=1)
    print('{} ECGs with all 0/NaN measurements'.format(to_excl.sum()))
    data = data[~to_excl]
    
    # print('Number of samples: {}'.format(data.shape[0]))
    print('Number of ECGs: {}'.format(data.ecg_id.nunique()))
    print('Number of patients: {}'.format(data.patient_id.nunique()))    

    # Tabular preprocessing
    tab_pipe = joblib.load(args.tabular_pipeline_path)
    tabular, _ = tabular_transformer(data[TABULAR_COLS], fit_yn=False, pipe=tab_pipe)
    
    # Waveform preprocessing
    waveforms, _ = extract_waveform(
        ecg_list=data['processed_npy_path'].tolist(), 
        norm_file=args.waveform_params_path, 
        n_jobs=args.n_jobs)

    # save datasets for inference
    data.to_parquet(os.path.join(args.out_dir, 'ecg_metadata_final.parquet'))
    np.save(os.path.join(args.out_dir, 'tabular_features.npy'), tabular.to_numpy())
    np.save(os.path.join(args.out_dir, 'waveforms.npy'), waveforms)

