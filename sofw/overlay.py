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

from sofw.utils import set_path, get_ts_from_filename, str2bool

def parse_args():
    usage = 'usage: generate the overlayed audio file from the time-scaled audio segments'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--ori-seg-dir', type=str, help='dir for the original L1 audio segments')
    parser.add_argument('--seg-dir', type=str, help='dir for the time scaled and shifted L2 audio segments')
    parser.add_argument('--out-dir', type=str, help='output dir of the overlayed audio file')
    parser.add_argument('--bg-audiofile', type=str, help='L1 background audio file')
    parser.add_argument('--vc-audiofile', type=str, help='L1 vocal audio file')
    parser.add_argument('--dur-lim', type=int, help='duration to be processed in minutes')
    parser.add_argument('--out-file', type=str, help='output overlayed audio file')
    parser.add_argument('--r', type=float, help='ratio of the reduced volumne of the background audio vs full volume ' + \
        '(1: no reduction, 0: reduce completly)')
    parser.add_argument('--with-res', type=str2bool, nargs='?', const=True,
        default=False, help='true if include the residules from the L1 vocal audio file')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    # voice = 'dmytro'
    # stress = 'dictionary'

    # args = argparse.ArgumentParser()
    # args.out_dir = os.path.join(work_dir, 'outputs', 'sofw', 'demo', recording_id, f'{voice}-{stress}', 'media')
    # args.ori_seg_dir = os.path.abspath(os.path.join(work_dir, 'data', recording_id, 'segments'))
    # args.seg_dir = os.path.abspath(os.path.join(work_dir, os.pardir, 'free-vc', 'outputs', recording_id, f'freevc-24_{voice}-{stress}_scaled'))
    # args.dur_lim = -1 # -1 means using the entire duration
    # args.with_res = True

    # # example 1: vocal-ukr + mixed-eng (30%)
    # data_dir = os.path.join(work_dir, 'data', recording_id, 'media')
    # args.bg_audiofile = os.path.join(data_dir, f'{recording_id}.wav')
    # args.vc_audiofile = os.path.join(args.out_dir, f'{recording_id}_vocals.wav')
    # args.out_file = os.path.join(args.out_dir, f'{recording_id}_bg+bivocals.wav')
    # args.r = 0.3 # reduce the background volume to 30%

    # # example 2: vocal-ukr + bg (100%)
    # args.bg_audiofile = os.path.join(args.out_dir, f'{recording_id}_accompaniment.wav')
    # args.vc_audiofile = os.path.join(args.out_dir, f'{recording_id}_vocals.wav')
    # args.out_file = os.path.join(args.out_dir, f'{recording_id}_bg+ukr.wav')
    # args.r = 1.0 # no volume reduction for the background

    # check dir/file existence
    assert os.path.isdir(args.ori_seg_dir), f'original segment dir: {args.ori_seg_dir} does not exist!'
    assert os.path.isdir(args.seg_dir), f'segment dir: {args.seg_dir} does not exist!'
    assert os.path.isfile(args.bg_audiofile), f'background audio file: {args.bg_audiofile} does not exist!'
    assert os.path.isfile(args.vc_audiofile), f'background audio file: {args.vc_audiofile} does not exist!'

    # localize arguments
    ori_seg_dir = args.ori_seg_dir
    seg_dir = args.seg_dir
    out_dir = args.out_dir
    out_file = args.out_file
    bg_audiofile = args.bg_audiofile
    vc_audiofile = args.vc_audiofile
    dur_lim = args.dur_lim
    r = args.r
    with_res = args.with_res

    # print arguments
    print(f'ori seg dir: {ori_seg_dir}')
    print(f'seg dir: {seg_dir}')
    print(f'out dir: {out_dir}')
    print(f'out file: {out_file}')
    print(f'background audio: {bg_audiofile}')
    print(f'vocal audio: {vc_audiofile}')
    print(f'duration limit: {dur_lim} min')
    print(f'background volume reduction factor: {r}')
    print(f'with residule: {with_res}')

    # get recording id
    recording_id = os.path.splitext(os.path.basename(bg_audiofile))[0]
    print(f'recording id: {recording_id}')

    # set dir
    out_dir = os.path.dirname(out_file)
    set_path(out_dir, verbose=True)

    # get the original audio segments
    ori_seg_audiofiles = sorted(glob.glob(os.path.join(ori_seg_dir, '*.wav')))
    num_ori_segments = len(ori_seg_audiofiles)
    print(f'# of original segments: {num_ori_segments}')

    # get the audio segments
    seg_audiofiles = sorted(glob.glob(os.path.join(seg_dir, '*.wav')))
    num_segments = len(seg_audiofiles)
    print(f'# of segments: {num_segments}')

    # sanity check: number of audio segments
    assert num_ori_segments == num_segments, '# of audio segments from input and output do not match!'
    del num_ori_segments

    # load L1 vocal audio file
    y_L1_vc, sr0 = librosa.load(vc_audiofile, sr=None)
    L_L1 = len(y_L1_vc)

    # supress the vocal to 0s
    y_L1_res = y_L1_vc
    for i in range(num_segments):
        _, start_time0, end_time0 = get_ts_from_filename(ori_seg_audiofiles[i])
        _, start_time1, end_time1 = get_ts_from_filename(seg_audiofiles[i])
        start_time = min(start_time0, start_time0)
        end_time = max(end_time1, end_time1)
        start_idx = int(start_time*sr0)
        end_idx = int(end_time*sr0)
        assert end_idx < L_L1, f'{i}/{num_segments}: segment end-time exceed the signal boundary'
        y_L1_res[start_idx:end_idx] = 0

    # write the residual audio file in eng
    out_file_res = os.path.join(out_dir, f'{recording_id}_res_eng.wav')
    sf.write(out_file_res, y_L1_res, sr0)
    print(f'wrote {out_file_res}')

    # get sampling rate
    _, sr1 = librosa.load(seg_audiofiles[0], sr=None)

    # get the duration (in seconds) of the background file
    dur_total = librosa.get_duration(path=bg_audiofile)
    print(f'total duration of the background audio file {bg_audiofile}: {dur_total:.2f} seconds')

    # create base signal with silence at length of dur_lim min
    if dur_lim == -1:
        L_L2 = int(np.ceil(dur_total * sr1))
    else:    
        L_L2 = int(np.ceil(min(dur_lim*60, dur_total) * sr1)) # base signal sample length
    dur_lim_sec = round(L_L2/sr1, 2)
    y_L2 = np.zeros(L_L2)

    # add in segments into the base signal
    diff_abs0 = int(0.005 * 2 * sr1) + 1 # the max difference due to start time and end time round error
    for i in range(num_segments):
        y, sr = librosa.load(seg_audiofiles[i], sr=None)
        nsamples = len(y)
        assert sr == sr1, f'{i}/{num_segments}: sampling rate inconsistent'
        _, start_time, end_time = get_ts_from_filename(seg_audiofiles[i])
        start_idx = int(start_time*sr1)
        end_idx = int(end_time*sr1)
        nsamples2 = end_idx - start_idx
        assert end_idx < L_L2, f'{i}/{num_segments}: segment end-time exceed the signal boundary'
        diff_abs = np.abs(nsamples-nsamples2)
        assert  diff_abs <= diff_abs0, \
            f'{i}/{num_segments}: segment sample length ({nsamples}) and the allocated sample length ' + \
            f'({nsamples2}) should differ no more than {diff_abs0}, but now {diff_abs}'
        y_L2[start_idx:start_idx+nsamples] = y

    # write pure vocal audio file in ukr
    out_file_vocals = os.path.join(out_dir, f'{recording_id}_vocals_ukr.wav')
    sf.write(out_file_vocals, y_L2, sr1)
    print(f'wrote {out_file_vocals}')

    # read the L1 vocal audio file (up to the duration limist)
    y_L1_res, _ = librosa.load(out_file_res, sr=sr1, mono=True, offset=0.0, duration=dur_lim_sec)
    L_L1_res = len(y_L1_res)

    # read the L1 background audio file (up to the duration limit)
    y_L1_bg, _ = librosa.load(bg_audiofile, sr=sr1, mono=True, offset=0.0, duration=dur_lim_sec)
    L_L1_bg = len(y_L1_bg)

    # sanity check on the audio track lengths
    assert L_L1_res == L_L1_bg, f'check L_L1_res ({L_L1_res}) and ({L_L1_bg})'
    assert L_L1_bg == L_L2 or L_L1_bg - L_L2 == 1, f'check L_L2 ({L_L2}) and L1 ({L1})'

    # combine foreground (vocals) and background (music) with a reducing factor
    if with_res:
        y2 = y_L2 + (y_L1_res[:L_L2] + y_L1_bg[:L_L2]) * r
    else:
        y2 = y_L2 + y_L1_bg[:L_L2] * r

    # write out the overlayed signal
    sf.write(out_file, y2, sr1)
    print(f'wrote {out_file}')