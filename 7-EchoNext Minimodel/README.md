# EchoNext model inference 

# I. Input preparation

All xmls are expected in a flat folder in a single directory with `*.xml` file names along with a file named `labels.csv`. The names of the xml files will be condsidered the ecg_id. The labels.csv file should include a header and have one row for each file with a ecg_id column(xml file name without extension) followed by these 12 binary labels:
   - lvef_lte_45 
   - lvwt_gte_13
   - aortic_stenosis_moderate_severe
   - aortic_regurgitation_moderate_severe
   - mitral_regurgitation_moderate_severe
   - tricuspid_regurgitation_moderate_severe
   - pulmonary_regurgitation_moderate_severe
   - rv_systolic_dysfunction_moderate_severe
   - pericardial_effusion_moderate_large
   - pasp_gte_45
   - tr_max_gte_32
   - shd

   **Note**: If the xml data has already been preprocessed into numpy waveform and feature arrays, skip to step C. to run inference.

## II. Docker and All-in-One Command

### A. Building Docker file.

To build the image echonext_val, download this directory to target computer and run the follow from this directory:

```
    docker build -t echonext_val .
```

### B. Running Docker Image.

Running the docker image will run all 3 command bellow in order with the default parameters (see runall.sh script). The run command is  
```
    docker run -v /path/to/xml:/xml_input -v /path/to/processed_data:/processed_data -v /path/to/results:/results echonext_val
```


You will need 3 directories to mount:
   - `/xml_input` - The directory of your ECGs in XML format and the labels.csv file as detailed above. 
   - `/processed_data` - The directory where processed ECG data will be stored.
   - `/results` - The directory where inferences will be stored.

  >Note: to run performance metrics (auroc and auprc), the `/xml_input` folder should also have a `labels.csv` file that contains ground truth labels (11 component label + 1 composite SHD label) as well as a index column named 'ecg_id'. The 'ecg_id' is assumed to be the xml filename without extension (e.g. for a xml file 'MUSE0001.xml', the ecg_id is 'MUSE0001'). 

## III. Individual command reference
  
### A. xml file parsing

Run the follow command to parse xml files. 


```
    python parse_xml.py \
    \
    --xml_dir /path/to/directory/with/ecg/xml/files/ \
    --out_dir /path/to/save/processed/data/ \
    --n_jobs 10
```

 Input arguments:   
    `--xml_dir`: directory that contains ecg xml files for testing  
    `--n_jobs`: number of parallel processes, default 10  
    `--out_dir`: directory to save processed data, including: a. parquet file that contains ecg meta data and measurements, b. raw numpy arrays of the 12-lead ECG waveforms, c. processed numpy arrays of 12-lead waveforms after baseline wander removal  


### B. preprocessing

Preprocessing step includes: 
1. **ECG filtering** The following ECGs were excluded from further analysis: pediatric (age<18), poor quality, ventricular paced, missing sex/age, all of ECG measurements missing. A final parquet file is saved to use as the index file of the final test set. 
2. **Tabular feature standardization** 7 tabular features are normalized using pretrained parameters from the model train set. A numpy array of Nx7 is saved to use as input to the EchoNext model.  
3. **Waveform normalization** waveforms were normalized using pretrained parameters from model train set. All ECG waveforms are stacked up as a big numpy array (shape: Nx1x2500x12) to use as input to the model. 

Command: 
```
python preprocess.py \
\
    --ecgmeta_path /output/directory/from/previous/step/echonext_ecg_metadata.parquet \
    --out_dir /path/to/save/processed/data/ \
    --tabular_pipeline_path /path/to/pretrained/tabular/pipeline/tabular_transformer.joblib \
    --waveform_params_path /path/to/pretrained/waveform/normalization/params/waveform_normalization_params.json \
    --n_jobs 10
```

Input arguments:  
`--ecgmeta_path`: path to the ecg metadata parquet file generated during the parse_xml step. Default file name: echonext_ecg_metadata.parquet    
`--out_dir`: directory to save preprocessed data, including:    
* **ecg_metadata_final.parquet**, final ecg parquet file, this is the index file for the numpy arrays, i.e. tabular feature numpy array and stacked waveform numpy array
* **tabular_features.npy**, numpy array of tabular features from all test ECGs in the same order of the metadata index file  
* **waveforms.npy**, numpy array of waveforms from all test ECGs, in the same order of the metadata index file  

`--tabular_pipeline_path`: path to pretrained pipeline for processing tabular features, provided as part of the code package (filename: `tabular_transformer.joblib`)  
`--waveform_params_path`: path to pretrained waveform normalization parameters, provided as part of the code package (filename: `waveform_normalization_params.json`)  
`--n_jobs`: number of parallel processes, default: 10.   

### C. Inference

Run inference with best model checkpoint to get prediction scores. 

Command: 
```
python cradlenet/scripts/inference/ecg_tabular.py \
  \
    --features_path /path/to/waveform/numpy/array/waveforms.npy \
    --tabular_path /path/to/tabular/feature/numpy/array/tabular_features.npy \
  \
    --legacy echonext \
    --batch_size 256 \
    --num_workers 2 \
    --write_outputs \
    --output_dir /directory/to/save/prediction/scores/ \
    --num_classes 12 \
    --len_tabular_feature_vector 7 \
    --filter_size 16 \
    --binary \
  \
    --checkpoint /path/to/model/checkpoint.pt
```

Input arguments:  
`--features_path`: path to the waveforms.npy file created by the preprocessing step  
`--tabular_path`: path to the tabular_features.npy file created by the preprocessing step   
`--output_dir`: output directory to save prediction scores  
`--checkpoint`: path to provided model weights. default: `models/echonext_multilabel_minimodel/weights.pt`

Output: `${output_dir}/prediction_loop/probs.npy`   
Shape of the output: N x 12, with each column representing prediction scores for one of the 12 disease labels listed below (in the same order): 
 - lvef_lte_45
 - lvwt_gte_13
 - aortic_stenosis_moderate_severe
 - aortic_regurgitation_moderate_severe
 - mitral_regurgitation_moderate_severe
 - tricuspid_regurgitation_moderate_severe
 - pulmonary_regurgitation_moderate_severe
 - rv_systolic_dysfunction_moderate_severe
 - pericardial_effusion_moderate_large
 - pasp_gte_45
 - tr_max_gte_32
 - shd

>Note: shd: structural heart disease, this is the composite label of all other labels.


### D. Evaluate performance

Evaluate model performance and saves AUROC and AUPRC.   

Command: 

```
python run_metrics.py \
\
  --labels_path /path/to/ground/truth/labels.csv \ 
  --prediction_path /path/to/prediction/results/prediction_loop/probs.npy \
  --ecgmeta_path /path/to/ecg/metadata/file/ecg_metadata_final.parquet \
  --output_dir /results/
```

Input argument:  
`--labels_path`: csv file that contains ground truth labels. It is assumed that the data should have all 12 disease labels with no missingness. Index column is assumed to be 'ecg_id', which is the xml filename without extension (i.e. for xml file MUSE0001.xml, ecg_id would be MUSE0001). The dataframe is cross referenced with the ecg metadata file (ecg_metadata_final.parquet) to select appropriate samples and align with prediction scores.   
`--prediction_path`: npy file that contains prediction scores from previous step. Default path: /results/prediction_loop/probs.npy  
`--ecgmeta_path`: path to ecg metadata file generated during preprocessing step. Default path: /processed_data/ecg_metadata_final.parquet   
`--output_dir`: path to save the performance metrics. Default: /results

