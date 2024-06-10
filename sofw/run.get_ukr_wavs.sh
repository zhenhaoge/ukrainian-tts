#/bin/bash!
#
# run get_ukr_wavs.py and get_ukr_wavs.mms.py
#
# Zhenhao Ge, 2024-05-22

WORK_DIR=/home/users/zge/code/repo/ukr-tts
TRANS_DIR=/home/splola/kathol/SOFW/StaticVideos/scripts

recording_id=MARCHE_AssessmentTacticalEnvironment
trans_file=$TRANS_DIR/${recording_id}.ukr.cor.sentids

# option 1: espnet

recipe=espnet
model_path=$WORK_DIR/model/${recipe}
voice=oleksa # options: tetiana, mykyta, lada, dmytro, oleksa
device='cuda:1'

for stress in dictionary model; do
    output_path=$WORK_DIR/outputs/sofw/${recipe}/${recording_id}/${voice}-${stress}
    echo "output path: ${output_path}"

    python $WORK_DIR/sofw/gen_ukr_wavs.py \
        --trans-file $trans_file \
        --model-path $model_path \
        --output-path $output_path \
        --voice $voice \
        --stress $stress \
        --device $device

    output_scaled_path=${output_path}_scaled
    python $WORK_DIR/sofw/norm_spk_rate.py \
        --input-path $output_path \
        --output-path $output_scaled_path

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

    output_scaled_path=${output_path}_scaled
    python $WORK_DIR/sofw/norm_spk_rate.py \
        --input-path $output_path \
        --output-path $output_scaled_path

done



