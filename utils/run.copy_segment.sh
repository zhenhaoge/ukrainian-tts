# run copy_segment.py to copy multiple versions of audio segments to the destination folder,
# renaming with appendix, for demo
#
# Zhenhao Ge, 2024-06-21

WORK_DIR=/home/users/zge/code/repo/ukr-tts

recording_id=${1-'MARCHE_AssessmentTacticalEnvironment'}
voice=${2:-dmytro}
stress=${3:-dictionary}

spk_folder=${voice}-${stress}
out_path=${4:-$WORK_DIR/outputs/sofw/demo/${recording_id}/${spk_folder}/segments}

ori_path=$WORK_DIR/data/${recording_id}/segments
syn_path=$WORK_DIR/outputs/sofw/espnet/${recording_id}/${spk_folder}
converted_path=/home/users/zge/code/repo/free-vc/outputs/${recording_id}/freevc-24_${spk_folder}
scaled_path=${converted_path}"_scaled"
keywords='.16000,_new,_converted,_paired,_unpaired,_v1,_v2,_v2,_v4'

# print out arguments
echo "original segment path: ${ori_path}"
echo "synthesized segment path: ${syn_path}"
echo "converted segment path: ${converted_path}"
echo "scaled segment path: ${scaled_path}"
echo "output segment path: ${out_path}"
echo "keywords: ${keywords}"

python $WORK_DIR/utils/copy_segment.py \
    --ori-path $ori_path \
    --syn-path $syn_path \
    --converted-path $converted_path \
    --scaled-path $scaled_path \
    --out-path $out_path \
    --keywords $keywords