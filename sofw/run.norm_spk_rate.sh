#/bin/bash!
#
# run norm_spk_rate.py
#
# Zhenhao Ge, 2024-06-19

ROOT_DIR=$HOME/code/repo
WORK_DIR=$ROOT_DIR/ukr-tts

# elements in the input path
recording_id="MARCHE_AssessmentTacticalEnvironment"
voice=dmytro
stress=dictionary

# reference path (original english segments)
reference_path=$WORK_DIR/data/$recording_id/segments

# set input (original ukrainian tts segments)

# # option 1: time-scale the original ukrainian tts segments
# input_path=$WORK_DIR/outputs/sofw/espnet/$recording_id/${voice}-${stress}

# option 2: time-scale the voice-converted ukrainian tts segments
input_path=$ROOT_DIR/free-vc/outputs/$recording_id/freevc-24_${voice}-${stress}

# set output path (time-scaled ukrainian tts segemnts)
output_path=${input_path}_scaled

# sanity check
[ -d $input_path ] || echo "input path: $input_path does not exist!"

# print input arguments
echo "input path: $input_path"
echo "output path: $output_path"
echo "reference path: $reference_path"

python $WORK_DIR/sofw/norm_spk_rate.py \
    --input-path $input_path \
    --output-path $output_path \
    --reference-path $reference_path