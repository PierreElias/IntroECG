import numpy as np 
import pandas as pd 
import os 
import argparse 

from sklearn.metrics import roc_auc_score, average_precision_score


ap = argparse.ArgumentParser(description='report metrics of echonext model')
ap.add_argument(
    '--labels_path', 
    type=str, 
    default='/xml_input/labels.csv', 
    help='Path to labels and index metadata file'
)
ap.add_argument(
    '--ecgmeta_path', 
    type=str, 
    default='/processed_data/ecg_metadata_final.parquet', 
    help='Path to index metadata file'
)
ap.add_argument(
    '--prediction_path', 
    type=str, 
    default='/results/prediction_loop/probs.npy',
    help='path to prediction scores')
ap.add_argument(
    '--output_dir', 
    type=str, 
    default='/results/',
    help='path to save performance metrics')

args = ap.parse_args()


"""
The input labels file should have shape N x 13, where the 13 columns are: 

ecg_id, 
lvef_lte_45, 
lvwt_gte_13, 
aortic_stenosis_moderate_severe, 
aortic_regurgitation_moderate_severe, 
mitral_regurgitation_moderate_severe,
tricuspid_regurgitation_moderate_severe,
pulmonary_regurgitation_moderate_severe,
rv_systolic_dysfunction_moderate_severe,
pericardial_effusion_moderate_large,
pasp_gte_45,
tr_max_gte_32,
shd

Note:
ecg_id is assmed to be xml filename without extension, e.g. for a xml file MUSE001.xml, the ecg_id is MUSE001
the labels file is assumed to be a csv file and saved in the same directory of the xml files. 

"""

label_cols = [
    'lvef_lte_45',
    'lvwt_gte_13',
    'aortic_stenosis_moderate_severe',
    'aortic_regurgitation_moderate_severe',
    'mitral_regurgitation_moderate_severe',
    'tricuspid_regurgitation_moderate_severe',
    'pulmonary_regurgitation_moderate_severe',
    'rv_systolic_dysfunction_moderate_severe',
    'pericardial_effusion_moderate_large',
    'pasp_gte_45',
    'tr_max_gte_32',
    'shd']

index_df = pd.read_parquet(args.ecgmeta_path)

scores = np.load(args.prediction_path)
assert scores.shape[1] == 12, 'prediction scores need to have 12 columns!'
scores = pd.DataFrame(scores, columns=label_cols)

assert scores.shape[0] == index_df.shape[0], 'length of predictions does not match index file!'

# load labels
labels = pd.read_csv(args.labels_path)
# keep valid samples
labels = index_df.set_index('ecg_id').join(labels.set_index('ecg_id'))[label_cols]
assert np.isnan(labels.to_numpy()).sum()==0, 'no missingness is allowed in labels'

print('Label prevalence')
print(labels.mean()*100)

res = []
for lbl in label_cols:
    auc = roc_auc_score(labels[lbl], scores[lbl])*100
    prc = average_precision_score(labels[lbl], scores[lbl])*100
    res.append([lbl, auc, prc])

res = pd.DataFrame(res, columns = ['Label', 'AUROC', 'AUPRC']).set_index('Label')
res.to_csv(os.path.join(args.output_dir, 'log_metrics.csv'))

print(res)