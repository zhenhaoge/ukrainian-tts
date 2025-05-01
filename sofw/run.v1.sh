#!/bin/bash
#
# master script to run video dubbing for the SOFW project
#
# version 1
# Zhenhao Ge, 2024-08-27

export https_proxy=http://10.16.0.132:8000
export http_proxy=http://10.16.0.132:8000

# use conda envionment 'espnet'
source /home/users/zge/.zshrc
conda activate espnet

# set configuration
recording_id=MARCHE_AssessmentTacticalEnvironment
# recording_id=F-16ViperCockpitTour
# recording_id=F-16ViperWalkaround
voice=dmytro
stress=dictionary

# set basic dirs (not dependent on config variables)
HOME_DIR=/home/users/zge
WORK_DIR=$HOME_DIR/code/repo/ukr-tts
TRANS_DIR=/home/splola/kathol/SOFW/StaticVideos/data/corrections
FREEVC_DIR=${HOME_DIR}/code/repo/free-vc
SPLEETER_DIR=${HOME_DIR}/code/repo/spleeter

# set additional dirs
DATA_DIR=$WORK_DIR/data/${recording_id}
DEMO_DIR=$WORK_DIR/outputs/sofw/demo/${recording_id}/${voice}-${stress}
MEDIA_DIR=$DEMO_DIR/media

# set input (existing) files
VIDEO_FILE_ORI=$DATA_DIR/media/${recording_id}.mp4
TXT_FILE_L1=$TRANS_DIR/${recording_id}-ASRcorrected1.v1.eng.sentids
TXT_FILE_L2=$TRANS_DIR/${recording_id}-ASRcorrected1.v1.ukr.sentids

# check file existence
[ -f $VIDEO_FILE_ORI ] || (echo "original video file: $VIDEO_FILE_ORI does not exist!" && exit 1)
[ -f $TXT_FILE_L1 ] || (echo "L1 text file: $TXT_FILE_L1 does not exist!" && exit 1)
[ -f $TXT_FILE_L2 ] || (echo "L2 text file: $TXT_FILE_L2 does not exist!" && exit 1)

# set additional files
AUDIO_FILE_ORI=${DATA_DIR}/media/${recording_id}.wav

# generate the original audio file if it does not exist
if [ ! -f $AUDIO_FILE_ORI ]; then
    echo "extracting audio from video $VIDEO_FILE_ORI ..."
    ffmpeg -i $VIDEO_FILE_ORI -vn $AUDIO_FILE_ORI # extract audio
    # ffmpeg -i $AUDIO_FILE_ORI # used to check audio file info
    echo "extracted audio file: $AUDIO_FILE_ORI"
else
    echo "$AUDIO_FILE_ORI already exist."
fi

# step 1: separate the source audio into background and vocals
echo "step 1: separating the source audio into background and vocals ..."
bash $WORK_DIR/sofw/run.spleeter.sh $SPLEETER_DIR $AUDIO_FILE_ORI $MEDIA_DIR

# step 2: generate the source audio segments (used as reference segments in voice conversion)
echo "step 2: generating the source audio segments ..."
wav_file=$MEDIA_DIR/${recording_id}_vocals.wav
txt_file=$TXT_FILE_L1
out_path=$DATA_DIR/segments
python $WORK_DIR/utils/extract_segment.py \
    --wav-file ${wav_file} \
    --txt-file ${txt_file} \
    --out-path ${out_path}

# step 3: generate the target audio segments using ukr-tts
echo "step 3: generating the target audio segments ..."
trans_file1=$TXT_FILE_L1
trans_file2=$TXT_FILE_L2
model_path=$WORK_DIR/model/espnet
output_path=$WORK_DIR/outputs/sofw/espnet/${recording_id}/${voice}-${stress}
device='cuda:1'
python $WORK_DIR/sofw/gen_ukr_wavs.py \
    --trans-file1 ${trans_file1} \
    --trans-file2 ${trans_file2} \
    --model-path ${model_path} \
    --output-path ${output_path} \
    --voice ${voice} \
    --stress ${stress} \
    --device ${device}

# switch to another environment for FreeVC
conda activate style

# step 4: prepare the text file for voice conversion using FreeVC
echo "step 4: preparing the text file for voice conversion using FreeVC ..."
src_path=$WORK_DIR/outputs/sofw/espnet/${recording_id}/${voice}-${stress}
tgt_path=$DATA_DIR/segments
txt_file=$FREEVC_DIR/txtfiles/${recording_id}_${voice}-${stress}.txt
python $FREEVC_DIR/scripts/prep_txtfile.py \
    --src-path ${src_path} \
    --tgt-path ${tgt_path} \
    --txt-file ${txt_file}

# step 5: run FreeVC voice conversion to generate the converted audio segments
echo "step 5: converting the translated audio segments to the target speaker voice ..."
device=1 # GPU device id
bash $FREEVC_DIR/run.convert.sh $FREEVC_DIR ${recording_id} ${voice} ${stress} ${device}

# switch back to the original environment after voice conversion
conda activate espnet

# step 6: time scale and shift the translated audio segments
echo "step 6: time scaling and shifting the translated audio segments ..."
in_dir=$FREEVC_DIR/outputs/${recording_id}/freevc-24_${voice}-${stress}
out_dir=${in_dir}_scaled
ref_dir=$DATA_DIR/segments
meta_dir=$WORK_DIR/outputs/sofw/espnet/${recording_id}/${voice}-${stress}
audio_file=$AUDIO_FILE_ORI
speed_lim=1.5
python $WORK_DIR/sofw/scale_segment.py \
    --in-dir ${in_dir} \
    --out-dir ${out_dir} \
    --ref-dir ${ref_dir} \
    --meta-dir ${meta_dir} \
    --audio-file ${audio_file} \
    --speed-lim  ${speed_lim}

# step 7: generate the overlayed audio file
echo "step 7: generating the overlayed audio file ..."
ori_seg_dir=$DATA_DIR/segments
seg_dir=$FREEVC_DIR/outputs/${recording_id}/freevc-24_${voice}-${stress}_scaled
out_dir=$MEDIA_DIR
bg_audiofile=$MEDIA_DIR/${recording_id}_accompaniment.wav
vc_audiofile=$MEDIA_DIR/${recording_id}_vocals.wav
dur_lim="-1"
out_file=${out_dir}/${recording_id}_bg+ukr.wav
r="1.0"
with_res=true
python $WORK_DIR/sofw/overlay.py \
    --ori-seg-dir ${ori_seg_dir} \
    --seg-dir ${seg_dir} \
    --out-dir ${out_dir} \
    --bg-audiofile ${bg_audiofile} \
    --vc-audiofile ${vc_audiofile} \
    --dur-lim ${dur_lim} \
    --out-file ${out_file} \
    --r $r \
    --with-res ${with_res}

# step 8: combine the overlayed audio with video and add subtitle
echo "step 8: combing the overlayed audio with video and adding subtitle ..."
bash $WORK_DIR/sofw/combine_av.sh ${recording_id} ${voice} ${stress}  

# step 9: copy multiple versions of the audio segments to the same folder for comparison
echo "step 9: copying multiple vertsions of the audio segments to the same folder for comparison ..."
out_path=$DEMO_DIR/segments
bash $WORK_DIR/utils/run.copy_segment.sh ${recording_id} ${voice} ${stress} ${out_path}