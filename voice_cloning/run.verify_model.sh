#!/bin/bash
#
# Zhenhao Ge, 2024-06-06

WORK_DIR=$HOME/code/repo/ukr-tts
DATA_DIR=$HOME/data1/datasets/ukrainian-tts

# specify recording id
recording_id="MARCHE_AssessmentTacticalEnvironment"

# select voice and stress options
voice="lada" # options: tetiana, mykyta, lada, oleksa (dmytro is not available in ukrainian-tts-dataset)
stress_type="dictionary" # options: dictionary, model

# specify reference speaker path (to extract speaker embedding)

# option 1: from syn parallel audio files
speaker_path=$WORK_DIR/outputs/sofw/espnet/$recording_id/${voice}-${stress_type}
appendix="paired"

# # option 2: # from original nonparallel audio files (need to sort and select based on duration)
# speaker_path=$DATA_DIR/$voice/dataset_${voice}_22khz
# appendix="unpaired"

# specify other arguments
cache_folder=$WORK_DIR/model/espnet
output_path=$WORK_DIR/outputs/sofw/espnet/$recording_id/${voice}-${stress_type}
trans_path='/home/splola/kathol/SOFW/StaticVideos/scripts'
trans_file=${trans_path}/${recording_id}.ukr.cor.sentids
source_path="speechbrain/spkrec-ecapa-voxceleb"
model_saved_path=$WORK_DIR/model/speechbrain/spkrec-ecapa-voxceleb
device="cuda:1"

python $WORK_DIR/vc/verify_model.py \
  --cache-folder $cache_folder \
  --stress-type $stress_type \
  --voice $voice \
  --speaker-path $speaker_path \
  --output-path $output_path \
  --trans-file $trans_file \
  --source-path $source_path \
  --model-saved-path $model_saved_path \
  --device $device \
  --appendix $appendix
