# generate the overlayed audio file from the time-scaled audio segments
#
# Zhenhao Ge, 2024-07-10

import os
from pathlib import Path
import argparse
import glob
import librosa
import soundfile as sf
import numpy as np

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

from sofw.utils import set_path

def parse_args():
    usage = 'usage: generate the overlayed audio file from the time-scaled audio segments'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--seg-dir', type=str, help='dir for the scaled audio segments')
    parser.add_argument('--out-dir', type=str, help='output dir')
    parser.add_argument('--bg-audiofile', type=str, help='background audio file')
    parser.add_argument('--dur-lim', type=int, help='duration to be processed in minutes')
    parser.add_argument('--out-file', type=str, help='output audio file')
    parser.add_argument('--r', type=float, help='ratio of the reduced volumne of the background audio vs full volume ' + \
        '(1: no reduce, 0: reduce completly)')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # interactive mode
    recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    voice = 'dmytro'
    stress = 'dictionary'
    data_dir = os.path.join(work_dir, 'data', recording_id, 'media')
    args.out_dir = os.path.join(work_dir, 'outputs', 'sofw', 'demo', recording_id, f'{voice}-{stress}', 'media')
    args = argparse.ArgumentParser()
    args.seg_dir = os.path.abspath(os.path.join(work_dir, os.pardir, 'free-vc', 'outputs', recording_id, f'freevc-24_{voice}-{stress}_scaled'))
    args.dur_lim = -1 # -1 means using the entire duration

    # # example 1: vocal-ukr + mixed-eng (30%)
    # args.bg_audiofile = os.path.join(data_dir, f'{recording_id}.wav')
    # args.out_file = os.path.join(args.out_dir, f'{recording_id}_bg+bivocals.wav')
    # args.r = 0.3 # reduce the background volume to 30%

    # example 2: vocal-ukr + bg (100%)
    args.bg_audiofile = os.path.join(args.out_dir, f'{recording_id}_accompaniment.wav')
    args.out_file = os.path.join(args.out_dir, f'{recording_id}_bg+ukr.wav')
    args.r = 1.0 # no volume reduction for the background

    # check dir/file existence
    assert os.path.isdir(args.seg_dir), f'segment dir: {args.seg_dir} does not exist!'
    assert os.path.isfile(args.bg_audiofile), f'background audio file: {args.bg_audiofile} does not exist!'

    # localize arguments
    seg_dir = args.seg_dir
    out_dir = args.out_dir
    out_file = args.out_file
    bg_audiofile = args.bg_audiofile
    dur_lim = args.dur_lim
    r = args.r

    # print arguments
    print(f'seg dir: {seg_dir}')
    print(f'out dir: {out_dir}')
    print(f'out file: {out_file}')
    print(f'background audio: {bg_audiofile}')
    print(f'duration limit: {dur_lim} min')
    print(f'background volume reduction factor: {r}')

    # set dir
    out_dir = os.path.dirname(out_file)
    set_path(out_dir, verbose=True)

    # get the audio segments
    seg_audiofiles = sorted(glob.glob(os.path.join(seg_dir, '*.wav')))
    nsegments = len(seg_audiofiles)
    print(f'# of segments: {nsegments}')

    # get sampling rate
    _, sr0 = librosa.load(seg_audiofiles[0], sr=None)

    # get the duration (in seconds) of the background file
    dur_total = librosa.get_duration(path=bg_audiofile)

    # create base signal with silence at length of dur_lim min
    if dur_lim == -1:
        L0 = int(np.ceil(dur_total * sr0))
    else:    
        L0 = int(np.ceil(min(dur_lim*60, dur_total) * sr0)) # base signal sample length
    dur_lim_sec = round(L0/sr0, 2)
    y0 = np.zeros(L0)

    # add in segments into the base signal
    diff_abs0 = int(0.005 * 2 * sr0) + 1 # the max difference due to start time and end time round error  
    for i in range(nsegments):
        y, sr = librosa.load(seg_audiofiles[i], sr=None)
        nsamples = len(y)
        assert sr == sr0, f'{i}/{nsegments}: sampling rate inconsistent'
        parts = os.path.splitext(os.path.basename(seg_audiofiles[i]))[0].split('_')
        idx = int(parts[0])
        start_time = round(float(parts[1]), 2)
        end_time = round(float(parts[2]), 2)
        start_idx = int(start_time*sr0)
        end_idx = int(end_time*sr0)
        nsamples2 = end_idx - start_idx
        assert end_idx < L0, f'{i}/{nsegment}: segment end-time exceed the singal boundary'
        diff_abs = np.abs(nsamples-nsamples2)
        assert  diff_abs <= diff_abs0, \
            f'{i}/{nsegments}: segment sample length ({nsamples}) and the allocated sample length ' + \
            f'({nsamples2}) should differ no more than {diff_abs0}, but now {diff_abs}'
        y0[start_idx:start_idx+nsamples] = y

    # write pure vocal audio file in ukr
    out_file_vocals = os.path.join(out_dir, f'{recording_id}_vocals_ukr.wav')
    sf.write(out_file_vocals, y0, sr0)
    print(f'wrote {out_file_vocals}')

    # read the background file (up to the duration limit)
    y1, _ = librosa.load(bg_audiofile, sr=sr0, mono=True, offset=0.0, duration=dur_lim_sec)
    L1 = len(y1)
    assert L1 == L0 or L1-L0 == 1, f'check L0 ({L0}) and L1 ({L1})'

    # combine foreground (vocals) and background (music) with a reducing factor
    y2 = y0 + y1[:L0] * r

    # write out the overlayed signal
    sf.write(out_file, y2, sr0)
    print(f'wrote {out_file}')