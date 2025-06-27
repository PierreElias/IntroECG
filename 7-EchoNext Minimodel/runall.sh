#! /bin/sh
MODEL_DIR=${1:-models/echonext_multilabel_minimodel}

CHECKPOINT="$MODEL_DIR/weights.pt"
TABULAR_PIPELINE="$MODEL_DIR/tabular_transformer.joblib"
WAVEFORM_PARAMS="$MODEL_DIR/waveform_normalization_params.json"

python parse_xml.py \
    --xml_dir /xml_input \
    --out_dir /processed_data \
    --n_jobs 40

python preprocess.py \
    --ecgmeta_path /processed_data/echonext_ecg_metadata.parquet \
    --out_dir /processed_data \
    --tabular_pipeline_path "$TABULAR_PIPELINE" \
    --waveform_params_path "$WAVEFORM_PARAMS" \
    --n_jobs 20

python cradlenet/scripts/inference/ecg_tabular.py \
  --features_path /processed_data/waveforms.npy \
  --tabular_path /processed_data/tabular_features.npy \
  --legacy echonext \
  --batch_size 256 \
  --num_workers 2 \
  --write_outputs \
  --output_dir /results \
  --num_classes 12 \
  --len_tabular_feature_vector 7 \
  --filter_size 16 \
  --binary \
  --checkpoint "$CHECKPOINT"

python run_metrics.py \
  --ecgmeta_path /processed_data/ecg_metadata_final.parquet \
  --labels_path /xml_input/labels.csv \
  --prediction_path /results/prediction_loop/probs.npy \
  --output_dir /results
