# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:44:59 2018
Updated: Fri Oct 5 9:00 2018
Updated: Tue Dec 10 2019
Updated: Sat June 13 2020 (by Vijay Rajaram and Pierre Elias)
@author: Sushravya Raghunath, David vanMaanen
This step is processing traces in ECG PDFs into a dataframe.
The final dataframe will have the following columns:
['PT_MRN','TEST_ID','filename','lead','signal','y_calibration','x_spacing']
PT_MRN       : MRN of patient
TEST_ID      : TestID of the ECG
filename     : ECG PDF filename
lead         : leadname of the traces in the row
signal       : resampled signal
y_calibration: the calibration factor applied while resampling
x_spacing    : minimum time resolution between samples
Notes:
- calibration blip: short signal at the beginning of the signal traces for the leads.
- If scale_x or scale_y is 0, likely the calibration blip shape was not typical.
- The signal values correspond to the x and y positions from the PDF. This is subtracted to create a 0 baseline for the signal.

To run in bash $python ecg_pdf_to_dataframe.py <name_of_directory_with_ECG_PDFs>
"""

import asyncio
from tempfile import mkstemp
import os
import subprocess
from xml.dom.minidom import parse
import numpy as np
import pandas as pd
import json
from scipy.interpolate import interp1d
from tqdm import tqdm
from multiprocessing import Pool
import sys
from glob import glob

def find_pdf_files(input_dir):
  exts = ('pdf', 'PDF') # the tuple of file types
  files = []
  for ext in exts:
    files.extend(glob(os.path.join(input_dir, "*.{}".format(ext))))
  return files

def resample_core(signalx, signaly, sqlen, minspacing):
    '''
    Input:
        signalx : the x-axis data points
        signaly : the signal at signalx points
        sqlen   : The number of points the resampled signal
        minspacing: The minimum spacing between two points
    Output:
        newy    : The resampled signal of sqlen # of points with spacing of minspacing
    This function resamples the signal for the given minspacing to the given number of sample points.
    '''
    f = interp1d(signalx, signaly)
    minnum = len(np.arange(signalx[0], signalx[-1], minspacing))
    if minnum > sqlen:
        raise Exception ('Minnum (', minnum,') greater than num (',sqlen,')')
    xx = np.linspace(signalx[0], signalx[-1], sqlen)
    newy = f(xx)
    return(newy)

async def convert_pdf_to_svg(fname, outname) -> int:
    '''
    Input:
         fname   : PDF file name
         outname : SVG file name
    Output:
         outname : return outname (file saved to disk)
    This will convert PDF into SVG format and save it in the given outpath.
    '''
    cmd = f'pdftocairo -svg "{fname}" "{outname}"'
    proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    print("Started: %s, pid=%s" % (cmd, proc.pid), flush=True)
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        if stdout:
            print(stdout.decode())
        if stderr:
            print(f'Error in converting {fname} to SVG:\n{stderr.decode()}')
    return proc.returncode

def process_svg_to_pd_perdata(svgfile, pdffile=None):
    '''
    Input:
    svgfile - datapath for svg file
    Output (returns):
    data    : data for 12 leads(available 15 or 12 traces), scale_vales and resolution units in a pandas dataframe
    Hard coded values :
    1) length of signal = 6 is assumed to be the calibration tracing at the beginning of the
    trace (by experiment)
    '''
    columnnames = np.array(['I', 'II','III','aVR','aVL','aVF','V1','V2','V3','V4', \
                     'V5', 'V6', 'V1L','IIL','V5L'])
    doc = parse(svgfile)
    if pdffile is None:
        strn = 'FILENAME_NA'
    else:
        strn = os.path.splitext(os.path.basename(pdffile))[0]
    data = pd.DataFrame(columns = ['PT_MRN','TEST_ID','filename','lead','x','y']) #,'scale_x','scale_y'])
    a = 0
    spacingvals = []
    scale_vals = []
    try:
        siglen = []
        for path in doc.getElementsByTagName('path'):
            tmp = path.getAttribute('d')
            tmp_split = tmp.split(' ')
            signal_np = np.asarray([float(x) for x in tmp_split if (x != 'M' and x != 'L' and x != 'C' and x != 'Z' and x != '')])
            signalx = signal_np[0::2]
            signaly = signal_np[1::2]
            siglen.append(len(signalx))
        siglen = np.array(siglen)
        # these are the calibration signals
        cali6sigs = np.where(siglen == 6)[0]
        minposcali = np.min(cali6sigs)
        tmpstart = list(range(minposcali, len(siglen)))
        last15sigs = np.array(list(set(tmpstart)- set(cali6sigs)))
        # index for leads
        a = 0
        for ind, path in enumerate(doc.getElementsByTagName('path')):
            if ind in last15sigs:
                if a > 14:
                    continue
                tmp = path.getAttribute('d')
                tmp_split = tmp.split(' ')
                signal_np = np.asarray([float(x) for x in tmp_split if (x != 'M' and x != 'L' and x != 'C' and x != 'Z' and x != '')])
                signalx = signal_np[0::2]
                signaly = signal_np[1::2]
                # expect the name of the file to be ptmrn_testid format.
                tmp = strn.split('_')
                try:
                    pid, testid = tmp[0], tmp[1]
                except:
                    pid = tmp[0]
                    testid = tmp[0]
                data.loc[data.shape[0]] = [pid, testid, strn, columnnames[a],signalx,signaly]
                spacingx = [t -s for s,t in zip(signalx, signalx[1:])]
                spacingvals.append(np.min(spacingx))
                a += 1
            elif ind in cali6sigs:
                tmp = path.getAttribute('d')
                tmp_split = tmp.split(' ')
                signal_np = np.asarray([float(x) for x in tmp_split if (x != 'M' and x != 'L' and x != 'C' and x != 'Z' and x != '')])
                signalx = signal_np[0::2]
                signaly = signal_np[1::2]
                scale_vals.append([np.min(signaly), np.max(signaly)])
        if len(scale_vals) == 0:
            data = None
            return data
        sx = [x[0] for x in scale_vals]
        sy = [x[1] for x in scale_vals]
        startloc = [d[0] for d in data.x.values]
        leads_ip = len(startloc)
        a = np.sum(startloc[0:3] == startloc[0])
        b = np.sum(startloc[3:6] == startloc[3])
        c = np.sum(startloc[6:9] == startloc[6])
        d = np.sum(startloc[9:12] == startloc[9])
        if data.shape[0] == 15:
            e = np.sum(startloc[12:15] == startloc[12])
            checkrhs = [3,3,3,3,3]
            checklhs = [a,b,c,d,e]
            assert checklhs == checkrhs
            scale_x = [sx[0:3],sx[0:3],sx[0:3],sx[0:3], sx[3:6]]
            scale_y = [sy[0:3],sy[0:3],sy[0:3],sy[0:3], sy[3:6]]
        elif data.shape[0] == 12:
            checkrhs = [3,3,3,3]
            checklhs = [a,b,c,d]
            assert checklhs == checkrhs
            scale_x = [sx[0:3],sx[0:3],sx[0:3],sx[0:3]]
            scale_y = [sy[0:3],sy[0:3],sy[0:3],sy[0:3]]
        else:
            data=None
            return data
        scale_x = [y for x in scale_x for y in x]
        data['scale_x'] = scale_x[0:data.shape[0]]
        scale_y = [y for x in scale_y for y in x]
        data['scale_y'] = scale_y[0:data.shape[0]]
        data['minspacing'] = spacingvals[0:data.shape[0]]
    except:
        data =  None
    return data

def process_resample_data(data):
    # This is a hard-coded specs for the data which is defined based on the signal sampling frequency and arrangement of the signals in the
    # PDF file and the extraction mechanism used in `process_svg_to_pd_perdata`
    config = {}
    config['minspacing'] = 5.0
    config['seqlen'] = [1250, 1250, 1250, 1250, 1250, 1250, 1250, 1250, 1250, 1250, 1250, 1250, 5000, 5000, 5000]
    config['leadnames'] = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6','V1L','IIL','V5L']
    leadnames = config['leadnames'][0:data.shape[0]]
    resampled_data = pd.DataFrame(columns = ['PT_MRN','TEST_ID','filename','lead','signal','y_calibration','x_spacing'])
    for lead_iter in range(len(leadnames)):
        lead_data = data.loc[data.lead == leadnames[lead_iter]]
        #print(lead_data)
        sqlen = config['seqlen'][lead_iter]
        signalx = lead_data.x.values[0]
        signaly = lead_data.y.values[0] - lead_data.scale_x.values[0]
        calibration_y = (lead_data.scale_y.values[0] - lead_data.scale_x.values[0])/1000
        signaly = signaly/calibration_y
        newy = resample_core(signalx, signaly, sqlen, config['minspacing'])
        newy = newy.astype(np.float32)
        resampled_data.loc[resampled_data.shape[0]] = [lead_data.PT_MRN.values[0],
                                                       lead_data.TEST_ID.values[0],
                                                       lead_data.filename.values[0],
                                                       leadnames[lead_iter],
                                                       newy, calibration_y,
                                                       lead_data.minspacing.values[0].round()]
    return resampled_data

async def get_data_from_PDF(pdf_path):
    '''
    Input:
         pdf_path : path to the pdf
    Output:
         resampled_data : The final datafrane which contains resampled signal with other data specs.
    '''
    try:
        if not os.path.exists('svgs'):
            os.makedirs('svgs')
        fp, svgfile = mkstemp(suffix='.svg', dir='svgs')
        os.close(fp)
        convertExitStatus = await convert_pdf_to_svg(pdf_path, svgfile)
        assert 0 == convertExitStatus, f'convert_pdf_to_svg failed with status {convertExitStatus}'
        data  = process_svg_to_pd_perdata(svgfile, pdf_path)
        # Make sure all the signals have same sampling rate - it should be unless I picked up wrong signal from SVG and called it a signal
        assert (data.minspacing.round() == data.minspacing[0].round()).all(), 'Sampling is different for different leads'
        resampled_data = process_resample_data(data)
        assert ((resampled_data.shape[0] == 12) | (resampled_data.shape[0] == 15)) == True, 'Number of signals extracted does not match 12 or 15'
        return resampled_data
    finally:
        if os.path.exists(svgfile):
            os.remove(svgfile)

def create_dataframe(pdf_path):
  try:
      if sys.version_info >= (3, 7):
          finaldataframe = asyncio.run(get_data_from_PDF(pdf_path))
      else:
          loop = asyncio.get_event_loop()
          finaldataframe = loop.run_until_complete(get_data_from_PDF(pdf_path))
      return finaldataframe
  except Exception as e:
      print('Error in PDF : {} with error {}'.format(pdf_path, e))

def expand_list(df, list_column, new_column): 
    lens_of_lists = df[list_column].apply(len)
    origin_rows = range(df.shape[0])
    destination_rows = np.repeat(origin_rows, lens_of_lists)
    non_list_cols = (
      [idx for idx, col in enumerate(df.columns)
       if col != list_column]
    )
    expanded_df = df.iloc[destination_rows, non_list_cols].copy()
    expanded_df[new_column] = (
      [item for items in df[list_column] for item in items]
      )
    expanded_df.reset_index(inplace=True, drop=True)
    return expanded_df

if __name__ == '__main__':
    current_dir = os.getcwd()
    print("args", sys.argv)
    # pdf_paths = [os.path.join(current_dir, sys.argv[1])] #if you want input to be a file
    pdf_paths = find_pdf_files(sys.argv[1]) #if you want input to be a directory
    print(pdf_paths)
    # PARALLEL
    with Pool(processes=16) as pool: #change processes based off CPU thread count 
        # use imap instead of imap_unordered to retain order (easier for printing purposes)
        dfs = [df for df in tqdm(pool.imap(create_dataframe, pdf_paths), total = len(pdf_paths))]
        for pdf_path, df in zip(pdf_paths, dfs):
          # print(pdf_path, "\n", df, "\n\n") #if you want to print each individual ecg df 
          filename_csv=str(df['filename'][0])+'.csv'
          filename_pkl = str(df['filename'][0])+'.pkl'
          # df.to_pickle(filename_pkl)
          expanded_df = expand_list(df, "signal", "signal") #this explodes the list in df['signal'], creating 30k rows per ECG
          expanded_df.to_csv(filename_csv, index=False)
          df['signal'] = str(df['signal'])
          df.to_csv('test_df.csv', index=False)

    # OR SEQUENTIAL
    # for pdf_path in tqdm(pdf_paths, total = len(pdf_paths)):
    #     df = create_dataframe(pdf_path)
    #     print(pdf_path, "\n", df, "\n\n")
