import os 
import glob
import re
import argparse
import xmltodict
import base64
import struct
import numpy as np
import pandas as pd
import scipy.signal
from multiprocessing.pool import Pool
from functools import partial

# input ECGs should not be poor quality or ventricular paced
# if the following finding statements are included in the diagnosis text, 
# then ECG is excluded from analysis

poor_data_quality_statements = [
    '*** POOR DATA QUALITY, INTERPRETATION MAY BE ADVERSELY AFFECTED']

ventricular_pacing_statements = [
    'ATRIAL-SENSED VENTRICULAR-PACED COMPLEXES',
    'ATRIAL-SENSED VENTRICULAR-PACED RHYTHM',
    'AV DUAL-PACED COMPLEXES',
    'AV DUAL-PACED RHYTHM',
    'AV SEQUENTIAL OR DUAL CHAMBER ELECTRONIC PACEMAKER',
    'BIVENTRICULAR PACEMAKER DETECTED',
    'ELECTRONIC VENTRICULAR PACEMAKER',
    'VENTRICULAR- PACED COMPLEXES',
    'VENTRICULAR-PACED RHYTHM',
    'WITH A DEMAND PACEMAKER']


def ecg_meta_from_xml(xml_dict):

    # dictionary for ecg meta data    
    ecg = {}

    try:
        ecg['acquisition_dttm'] = pd.to_datetime(
            xml_dict['TestDemographics']['AcquisitionDate'] + " " + 
            xml_dict['TestDemographics']['AcquisitionTime'])
    except: 
        ecg['acquisition_dttm'] = pd.NaT
            
    age = xml_dict['PatientDemographics'].get('PatientAge')
    if age is not None and age.isnumeric():
        ecg['age_at_ecg'] = float(age) if xml_dict['PatientDemographics'].get('AgeUnits') == 'YEARS' else 0
    else:
        ecg['age_at_ecg'] = np.nan
        
    ecg['sex'] = xml_dict['PatientDemographics'].get('Gender')

    # Variables used as input to the model
    
    ecg['atrial_rate'] = float(xml_dict['RestingECGMeasurements'].get('AtrialRate', np.nan))
    ecg['ventricular_rate'] = float(xml_dict['RestingECGMeasurements'].get('VentricularRate', np.nan))
    ecg['pr_interval'] = float(xml_dict['RestingECGMeasurements'].get('PRInterval', np.nan))
    ecg['qrs_duration'] = float(xml_dict['RestingECGMeasurements'].get('QRSDuration', np.nan))
    ecg['qt_corrected'] = float(xml_dict['RestingECGMeasurements'].get('QTCorrected', np.nan))

    # other columns of interest
    ecg['patient_id'] = xml_dict['PatientDemographics'].get('PatientID')
    ecg['last_nm'] = xml_dict['PatientDemographics'].get('PatientLastName')
    ecg['first_nm'] = xml_dict['PatientDemographics'].get('PatientFirstName')
    ecg['birth_dt'] = pd.to_datetime(xml_dict['PatientDemographics'].get('DateofBirth'))
    ecg['race'] = xml_dict['PatientDemographics'].get('Race')
    ecg['site_id'] = xml_dict['TestDemographics'].get('Site')
    ecg['site_nm'] = xml_dict['TestDemographics'].get('SiteName')
    ecg['location_id'] = xml_dict['TestDemographics'].get('Location')
    ecg['location_nm'] = xml_dict['TestDemographics'].get('LocationName')

    # exclusion statement
    try:
        txt = ' '.join([stmt['StmtText'] for stmt in xml_dict['Diagnosis']['DiagnosisStatement'] if stmt['StmtText']]) if isinstance(xml_dict['Diagnosis']['DiagnosisStatement'], list) else ''
        ecg['poor_data_quality_flag'] = 1 if '|'.join(poor_data_quality_statements) in txt else 0
        ecg['ventricular_pacing_flag'] = 1 if '|'.join(ventricular_pacing_statements) in txt else 0
    except:
        print('Unable to parse diagnose text')

    return ecg


def decode_ekg_muse_to_array(raw_wave, downsample = 1):
    """
    Ingest the base64 encoded waveforms and transform to numeric

    downsample: 0.5 takes every other value in the array. Muse samples at 500/s and the sample model requires 250/s. So take every other.
    """
    try:
        dwnsmpl = int(1//downsample)
    except ZeroDivisionError:
        print("You must downsample by more than 0")
    # covert the waveform from base64 to byte array
    arr = base64.b64decode(bytes(raw_wave, 'utf-8'))

    # unpack every 2 bytes, little endian (16 bit encoding)
    unpack_symbols = ''.join([char*int(len(arr)/2) for char in 'h'])
    byte_array = struct.unpack(unpack_symbols,  arr)
    return np.array(byte_array)[::dwnsmpl]


def extract_waveform_from_xml(xml_dict):

    """
    Extract ECG waveform from xml and save as numpy array with shape=[2500,12,1] ([time, leads, 1]).
    The voltage unit should be in 1 mv/unit and the sampling rate should be 250/second (total 10 second).
    The leads should be ordered as follow I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
    """

    #need to instantiate leads in the proper order for the model
    lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    """
    Each EKG will have this data structure:
    lead_data = {
        'I': np.array,
        'II': np.array,
        ...
    }
    """

    lead_data = dict.fromkeys(lead_order)
    waveform = xml_dict['Waveform']

    # take the second waveform
    wave = waveform[-1]
    for lead in wave['LeadData']:
        waveform_data = lead['WaveFormData']
        lead_id = lead['LeadID']
        sample_count = lead['LeadSampleCountTotal']
        if sample_count == '5000':
            waveform_data_array = decode_ekg_muse_to_array(
                waveform_data, downsample=0.5)
        elif sample_count == '2500':
            waveform_data_array = decode_ekg_muse_to_array(
                waveform_data, downsample=1)
        else:
            continue
        lead_data[lead_id] = waveform_data_array

    lead_data['III'] = (lead_data["II"] - lead_data["I"])
    lead_data['aVR'] = -(lead_data["I"] + lead_data["II"]) / 2
    lead_data['aVF'] = lead_data["II"] - lead_data["I"] / 2
    lead_data['aVL'] = lead_data["I"] - lead_data["II"] / 2

    # drops V3R, V4R, and V7 if it was a 15-lead ECG and ensure leads are in the correct order
    # lead_data = {k: lead_data[k] for k in lead_order}
    ekg_df = pd.DataFrame(lead_data)[lead_order]

    # expand dims to [time, leads, 1]
    ekg_array = np.expand_dims(ekg_df, axis=-1)

    # Here is a check to make sure all the model inputs are the right shape
    assert ekg_array.shape == (
        2500, 12, 1), "ekg_array is shape {} not (2500, 12, 1)".format(ekg_array.shape)
    return ekg_array


def baseline_wander_removal(data, sampling_frequency=250):
    if data.shape[0] != 12:
        data = np.transpose(data)

    processed_data = np.zeros(data.shape)
    for lead in range(data.shape[0]):
        # Baseline estimation
        win_size = int(np.round(0.2 * sampling_frequency)) + 1
        baseline = scipy.signal.medfilt(data[lead,:], win_size)
        win_size = int(np.round(0.6 * sampling_frequency)) + 1
        baseline = scipy.signal.medfilt(baseline, win_size)
        # Removing baseline
        filt_data = data[lead,:] - baseline
        processed_data[lead,:] = filt_data

    processed_data = np.expand_dims(processed_data.T, axis=-1)
    assert processed_data.shape == (2500, 12, 1), \
        'processed ekg array is shape {}, not (2500, 12, 1)'.format(processed_data.shape)

    return processed_data


def read_xml(xmlfile):
    try:
        with open(xmlfile, 'rb') as fd:
            xml_dict = xmltodict.parse(fd.read().decode('utf8'))
        
        xml_dict = xml_dict['RestingECG']
        failed = 0
    except:
        xml_dict = None
        failed = 1

    ecgid = re.sub('.xml', '', xmlfile.split(os.sep)[-1])

    obj = {'xml_path': xmlfile, 'ecg_id': ecgid, 'xml_dict': xml_dict, 'failed': failed}

    return obj


def parse_xml(xml_obj, npypath, processed_npypath, overwrite=True):
    
    ecg = ecg_meta_from_xml(xml_obj['xml_dict'])
    
    ecgid = xml_obj['ecg_id']
    npyfile = os.path.join(npypath, ecgid+'.npy')
    processed_npy_file = os.path.join(processed_npypath, ecgid+'.npy')

    # Before proceed, check if npy already exists (only proceed if not existing or overwrite=True)
    if os.path.exists(processed_npy_file) and os.path.exists(npyfile) and overwrite==False:
        print('Numpy array already exists. Skipping..')
        ecg['file_created'] = 1
        ecg['ecg_id'] = ecgid
        ecg['xml_path'] = xml_obj['xml_path']
        ecg['npy_path'] = npyfile
        ecg['processed_npy_path'] = processed_npy_file
        return ecg

    try:
        # extract waveforms
        waveform = extract_waveform_from_xml(xml_obj['xml_dict'])
        np.save(npyfile, waveform)
        # baseline wander removal
        processed = baseline_wander_removal(waveform.squeeze())
        np.save(processed_npy_file, processed)
        ecg['file_created'] = 1
    except:
        ecg['file_created'] = 0

    ecg['ecg_id'] = ecgid
    ecg['xml_path'] = xml_obj['xml_path']
    ecg['npy_path'] = npyfile
    ecg['processed_npy_path'] = processed_npy_file

    return ecg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='xml ecg report parser')
    ap.add_argument('--xml_dir', '-i', type=str, help='Path to ECG xml directory')
    ap.add_argument('--out_dir', '-o', type=str, help='Output data directory')
    ap.add_argument('--n_jobs', '-p', type=int, default=10, help='Number of parallel processes')
    
    args = ap.parse_args()

    npy_path = os.path.join(args.out_dir, 'npy')
    processed_npy_path = os.path.join(args.out_dir, 'npy_processed')

    os.makedirs(npy_path, exist_ok=True)
    os.makedirs(processed_npy_path, exist_ok=True)

    flist = glob.glob(args.xml_dir + '/*.xml')

    # read xml files
    with Pool(processes=args.n_jobs) as pm:
        xml_list = pm.map(read_xml, flist)
    
    xml_valid = [xml_obj for xml_obj in xml_list if xml_obj['failed']==0]
    pd.DataFrame(xml_list)[['xml_path', 'failed']].to_csv(
        os.path.join(args.out_dir, 'log_xml_read.csv'), index=False)
    
    # parse xml files
    with Pool(processes=args.n_jobs) as pm:
        ecgmeta = pm.map(
            partial(parse_xml, npypath=npy_path, processed_npypath=processed_npy_path, overwrite=True), 
            xml_valid)
    
    ecgmeta = pd.DataFrame(ecgmeta)
    ecgmeta.to_parquet(os.path.join(args.out_dir, 'echonext_ecg_metadata.parquet'))
    
