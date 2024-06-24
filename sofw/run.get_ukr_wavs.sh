#/bin/bash!
#
# run get_ukr_wavs.py and get_ukr_wavs.mms.py
#
# Zhenhao Ge, 2024-05-22

WORK_DIR=/home/users/zge/code/repo/ukr-tts
recording_id=MARCHE_AssessmentTacticalEnvironment

# # old version
# TRANS_DIR=/home/splola/kathol/SOFW/StaticVideos/scripts
# trans_file1=$TRANS_DIR/${recording_id}.eng.cor.sentids
# trans_file2=$TRANS_DIR/${recording_id}.ukr.cor.sentids

# updated version
TRANS_DIR=/home/splola/kathol/SOFW/StaticVideos/data/corrections
trans_file1=$TRANS_DIR/${recording_id}-ASRcorrected1.v1.eng.sentids
trans_file2=$TRANS_DIR/${recording_id}-ASRcorrected1.v1.ukr.sentids

# reference path for the original english segments (used in speaker duration normalization)
reference_path=$WORK_DIR/data/$recording_id/segments

# option 1: espnet

recipe=espnet
model_path=$WORK_DIR/model/${recipe}
voice=dmytro # options: tetiana, mykyta, lada, dmytro, oleksa
device='cuda:0'

for stress in dictionary model; do
    output_path=$WORK_DIR/outputs/sofw/${recipe}/${recording_id}/${voice}-${stress}
    echo "output path: ${output_path}"

    python $WORK_DIR/sofw/gen_ukr_wavs.py \
        --trans-file1 $trans_file1 \
        --trans-file2 $trans_file2 \
        --model-path $model_path \
        --output-path $output_path \
        --voice $voice \
        --stress $stress \
        --device $device

    # output_scaled_path=${output_path}_scaled
    # python $WORK_DIR/sofw/norm_spk_rate.py \
    #     --input-path $output_path \
    #     --output-path $output_scaled_path \
    #     --reference-path $reference_path

done

# option 2: mms-tts

# cpu only, no voice options

recipe=mms-tts
model_path=$WORK_DIR/model/${recipe}

for stress in dictionary model; do
    output_path=$WORK_DIR/outputs/sofw/${recipe}/${recording_id}/${stress}
    echo "output path: ${output_path}"

    python $WORK_DIR/sofw/gen_ukr_wavs.mms.py \
        --trans-file $trans_file \
        --model-path $model_path \
        --output-path $output_path \
        --stress $stress

    # output_scaled_path=${output_path}_scaled
    # python $WORK_DIR/sofw/norm_spk_rate.py \
    #     --input-path $output_path \
    #     --output-path $output_scaled_path \
    #     --reference-path $reference_path

done



