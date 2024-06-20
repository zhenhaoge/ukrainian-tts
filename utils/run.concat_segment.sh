#!/bin/bash
#
# Zhenhao Ge, 2024-06-11

WORK_DIR=$HOME/code/repo/ukr-tts

recording_id=MARCHE_AssessmentTacticalEnvironment
voice=lada
stress=dictionary
spk_folder=${voice}-${stress}
wav_path=$WORK_DIR/outputs/sofw/espnet/${recording_id}/${spk_folder}
concat_wavpath=$WORK_DIR/outputs/sofw/espnet/${recording_id}/${spk_folder}_all.wav
keywords='.16000,_new,_converted,_paired,_unpaired,'
speaker_embed="speechbrain/spkrec-xvect-voxceleb"

echo "wav path: $wav_path"
echo "concat wav path $concat_wavpath"

python utils/concat_segment.py \
    --wav-path $wav_path \
    --keywords $keywords \
    --concat-wavpath $concat_wavpath \
    --speaker-embed $speaker_embed
