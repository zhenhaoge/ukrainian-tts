# run copy_segment.py to copy multiple versions of audio segments to the destination folder,
# renaming with appendix, for demo
#
# Zhenhao Ge, 2024-06-21

WORK_DIR=$HOME/code/repo/ukr-tts

recording_id='MARCHE_AssessmentTacticalEnvironment'
voice=dmytro
stress=dictionary
spk_folder=${voice}-{stress}

ori_path=$WORK_DIR/data/${recording_id}/segments
syn_path=$WORK_DIR/outputs/sofw/espnet/${recording_id}/${spk_folder}
converted_path=$HOME/code/repo/free-vc/outputs/${recording_id}/freevc-24_${spk_folder}
scaled_path=${converted_path}+"_scaled"
out_path=$WORK_DIR/outputs/sofw/demo/${recording_id}/${spk_folder}
keywords='.16000,_new,_converted,_paired,_unpaired, _v1, _v2, _v2, _v4'

python $WORK_DIR/utils/copy_segment.py \
    --ori-path $ori_path \
    --syn-path $syn_path \
    --converted-path $converted_path \
    --scaled-path $scaled_path \
    --out-path $out_path \
    --keywords $keywords