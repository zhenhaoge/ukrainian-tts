#!/bin/bash
#
# Zhenhao Ge, 2024-06-11

WORK_DIR=$HOME/code/repo/ukr-tts

# set the input wav path (containing multiple audio segments to be voice converted)
recording_id='MARCHE_AssessmentTacticalEnvironment'
voice=dmytro
stress=dictionary
wav_folder=${voice}-${stress}
input_path=$WORK_DIR/outputs/sofw/espnet/${recording_id}/${wav_folder}

# set filtered keywords
# (used to filter out unrelated wav files with filtered keywords in the file name)
keywords='.16000,_converted,_ukr,_resampled,_paired,_unpaired'

# select one of the following version for the reference path

# # v1: convert to the english speaker's voice using paired speaker embedding
# reference_path=$WORK_DIR/data/${recording_id}/segments # dir to all segments
# appendix=v1

# # v2: convert to the english speaker's voice using averaged speaker embedding
# reference_path=$WORK_DIR/data/${recording_id}/segments_all.wav # path of concatenated segment
# appendix=v2

# # v3: convert to a ukrainian speaker's voice using paired speaker embedding
# ref_voice='lada'
# ref_stress='dictionary'
# ref_wav_folder=${ref_voice}-${ref_stress}
# reference_path=$WORK_DIR/outputs/sofw/espnet/${recording_id}/${ref_wav_folder}
# appendix=v3

# v4: convert to a ukrainian speaker's voice using paired speaker embedding
ref_voice='lada'
ref_stress='dictionary'
ref_wav_folder=${ref_voice}-${ref_stress}
reference_path=$WORK_DIR/outputs/sofw/espnet/${recording_id}/${ref_wav_folder}_all.wav
appendix=v4

# set the output path for the converted audio segments
# (currently, set to the same input wav path for easy comparison)
output_path=$input_path

# set the path to the speaker embedding model
speaker_embed="speechbrain/spkrec-xvect-voxceleb"

# set the max # of segments to be processed
# (set to a small number to do POC test)
num_lim=5

echo "input path: $input_path"
echo "keywords: $keywords"
echo "reference path: $reference_path"
echo "output path: $output_path"
echo "appendix: $appendix"
echo "speaker embedding model: $speaker_embed"
echo "num of max segments processed: $num_lim"

python $WORK_DIR/sofw/convert_segment.py \
    --input-path $input_path \
    --keywords $keywords \
    --reference-path $reference_path \
    --output-path $output_path \
    --appendix $appendix \
    --speaker-embed $speaker_embed \
    --num-lim $num_lim