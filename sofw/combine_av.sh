#!/bin/bash
#
# Zhenhao Ge, 2024-07-10

recording_id=MARCHE_AssessmentTacticalEnvironment
voice=dmytro
stress=dictionary

# set dirs
HOME_DIR=$HOME/code/repo/ukr-tts
DATA_DIR=$HOME_DIR/data/${recording_id}
SEG_DIR=$HOME/code/repo/free-vc/outputs/${recording_id}/freevc-24_${voice}-${stress}_scaled
OUT_DIR=$HOME_DIR/outputs/sofw/demo/${recording_id}/${voice}-${stress}
# VIDEO_DIR=$HOME/code/unclassfied/SOFW/kathol/StaticVideos/data/video

# check dir existence
[ -d $DATA_DIR ] || echo "data dir: $DATA_DIR does not exist!"
[ -d $SEG_DIR ] || echo "segment dir: $SEG_DIR does not exist!"
[ -d $OUT_DIR ] || echo "output dir: $OUT_DIR does not exist!"

# specify files
VIDEO_FILE_ORI=${DATA_DIR}/media/${recording_id}.mp4
VIDEO_FILE_SIL=${DATA_DIR}/media/${recording_id}_nosound.mp4
AUDIO_FILE_ORI=${DATA_DIR}/media/${recording_id}.wav
STT_FILE_IN=${SEG_DIR}/${recording_id}_meta.csv
STT_FILE_OUT=${SEG_DIR}/${recording_id}_ukr.srt
# AUDIO_FILE_OVL=${OUT_DIR}/media/${recording_id}_bg+bivocals.wav
# VIDEO_FILE_CMD=${OUT_DIR}/media/${recording_id}_bg+bivocals.mp4
# VIDEO_FILE_STT=${OUT_DIR}/media/${recording_id}_bg+bivocals_subtitled.mp4
AUDIO_FILE_OVL=${OUT_DIR}/media/${recording_id}_bg+ukr.wav
VIDEO_FILE_CMD=${OUT_DIR}/media/${recording_id}_bg+ukr.mp4
VIDEO_FILE_STT=${OUT_DIR}/media/${recording_id}_bg+ukr_subtitled.mp4

# check file existence
[ -f $VIDEO_FILE_ORI ] || echo "original video file: $VIDEO_FILE_ORI does not exist!"
[ -f $AUDIO_FILE_ORI ] || echo "original audio file: $AUDIO_FILE_ORI does not exist!"
[ -f $STT_FILE_IN ] || echo "subtitle input file: $STT_FILE_IN does not exist!"
[ -f $AUDIO_FILE_OVL ] || echo "overlayed audio file: $AUDIO_FILE_OVL does not exist!"

# print out files
echo "original video file: $VIDEO_FILE_ORI"
echo "silence video file: $VIDEO_FILE_SIL"
echo "original audio file: $AUDIO_FILE_ORI"
echo "subtitle input file: $STT_FILE_IN"
echo "subtitle output file: $STT_FILE_OUT"
echo "overlayed audio file: $AUDIO_FILE_OVL"
echo "combined video file: $VIDEO_FILE_CMD"
echo "subtitled video file: $VIDEO_FILE_STT"

# get video dur
video_dur_ori=$(ffmpeg -i $VIDEO_FILE_ORI 2>&1 | grep Duration | awk '{print $2}' | cut -d ',' -f 1)
echo "video duration: $video_dur_ori"

# get audio dur
audio_dur_ori=$(soxi -d $AUDIO_FILE_ORI)
audio_dur_overlayed=$(soxi -d $AUDIO_FILE_OVL)
echo "original audio duration $audio_dur_ori"
echo "overlayed audio duration $audio_dur_overlayed"

# extract video without audio from the original video
if [ ! -f $VIDEO_FILE_SIL ]; then
    ffmpeg -i $VIDEO_FILE_ORI -c copy -an $VIDEO_FILE_SIL
fi

# combine audio and video (no sound) files
ffmpeg -i $VIDEO_FILE_SIL -i $AUDIO_FILE_OVL -c:v copy -c:a aac $VIDEO_FILE_CMD
echo "$(basename $VIDEO_FILE_SIL) + $(basename $AUDIO_FILE_OVL) -> $(basename $VIDEO_FILE_CMD)"

# create the subtitle file
python sofw/prep_srt.py \
  --meta-file $STT_FILE_IN \
  --srt-file $STT_FILE_OUT

# add soft subtile to the video file
ffmpeg -i $VIDEO_FILE_CMD -i $STT_FILE_OUT -c copy -c:s mov_text -metadata:s:s:0 language=ukr $VIDEO_FILE_STT
echo "$(basename $VIDEO_FILE_CMD) + $(basename $STT_FILE_OUT) -> $(basename $VIDEO_FILE_STT)"