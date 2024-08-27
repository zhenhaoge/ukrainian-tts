# scale the translated audio segments to have the same total duration of the original segments
# adjust the start time and end time of the scaled segment to fit in the space, here are the strategies used
#  - use the context silence space with minimum gap
#  - scale more than the averge (up to the cap of 1.5x)
#  - shift the next segment for a delayed start time (offset)
#
# this script was developed similarly as ukr-tts/sofw/norm_spk_rate.py, but it more complicated
# norm_spk_rate.py just scale the segments with an overall scaling factor, but here it do other tricks to avoid
# overlap and still maintaining the syncness with the original segments
#
# Zhenhao Ge, 2024-07-05

import os
from pathlib import Path
import glob
import json
import numpy as np
import argparse
import librosa
import soundfile as sf

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

from sofw.utils import get_dur_from_meta, get_dur_from_file
from sofw.utils import get_ts_from_filename, find_bound, get_scaled_ts
from sofw.utils import dl2csv, tuple2csv, set_path, empty_dir
from audio import adjust_speed

keywords = '.16000,_converted,_ukr,_resampled,_paired,_unpaired'
keywords = keywords.split(',')

def filter_path(paths, keywords):
    for kw in keywords:
        paths = [f for f in paths if kw not in f]
    return paths

def parse_args():
    usage = 'usage: time-scale audio segments'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--in-dir', type=str, help='input dir of tts speech in its original speed')
    parser.add_argument('--out-dir', type=str, help='output dir of the output time-scaled tts speech')
    parser.add_argument('--ref-dir', type=str, help='reference dir of the real speech ' + \
        '(get actual duration to match in case there is no meta json file)')
    parser.add_argument('--meta-dir', type=str, help='meta dir where to get the meta data')    
    parser.add_argument('--audio-file', type=str, help='audio file to extract segments from')
    parser.add_argument('--speed-lim', type=float, default=1.5, \
        help='speed-up limit from original tts segments to scaled tts segments')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # work_dir = os.getcwd()
    # recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    # data_dir = os.path.join(work_dir, 'data', recording_id)
    # voice = 'dmytro'
    # stress = 'dictionary'
    # args = argparse.ArgumentParser()
    # # args.in_dir = os.path.join(work_dir, 'outputs', 'sofw', 'espnet', recording_id, '{}-{}'.format(voice, stress))
    # args.in_dir = os.path.abspath(os.path.join(work_dir, os.pardir, 'free-vc', 'outputs', recording_id, f'freevc-24_{voice}-{stress}'))
    # args.out_dir = args.in_dir + '_scaled'
    # args.ref_dir = os.path.join(data_dir, 'segments')
    # args.meta_dir = os.path.join(work_dir, 'outputs', 'sofw', 'espnet', recording_id, '{}-{}'.format(voice, stress))
    # args.audio_file = os.path.join(data_dir, 'media', f'{recording_id}.wav')
    # args.speed_lim = 1.5 # speed up to 1.5x from the syn segment to the scaled segment

    # check file/dir existence
    assert os.path.isdir(args.in_dir), f'input dir: {args.in_dir} does not exist!'
    assert os.path.isdir(args.ref_dir), f'reference dir: {args.ref_dir} does not exist!'
    assert os.path.isdir(args.meta_dir), f'meta dir: {args.meta_dir} does not exist!'
    assert os.path.isfile(args.audio_file), f'audio file: {args.audio_file} does not exist!'

    # create output dir
    set_path(args.out_dir, verbose=True)
    empty_dir(args.out_dir)

    # localize arguments
    in_dir = args.in_dir
    out_dir = args.out_dir
    ref_dir = args.ref_dir
    meta_dir = args.meta_dir
    audio_file = args.audio_file
    speed_lim = args.speed_lim

    # print arguments
    print(f'input dir: {in_dir}')
    print(f'output dir: {out_dir}')
    print(f'reference dir: {ref_dir}')
    print(f'meta dir: {meta_dir}')
    print(f'audio file: {audio_file}')
    print(f'speed lim: {speed_lim}x')

    # get the input audio files
    input_audiofiles = sorted(glob.glob(os.path.join(in_dir, '*.wav')))
    input_audiofiles = filter_path(input_audiofiles, keywords)
    nsegments = len(input_audiofiles)
    print(f'# of segments: {nsegments}')

    # get the reference audio files
    reference_audiofiles = sorted(glob.glob(os.path.join(ref_dir, '*.wav')))
    reference_audiofiles = filter_path(reference_audiofiles, keywords)
    assert len(reference_audiofiles) == nsegments, '# of segments in the input and reference dirs mis-match!'

    # get timestamp tuple list from the input audio filenames
    ts_lst = get_ts_from_filename(input_audiofiles)

    # get durations for the referenced and synthesized wavs from files directly
    durs_syn = get_dur_from_file(input_audiofiles)
    durs_ref = get_dur_from_file(reference_audiofiles)

    # get the total duration for the referenced and synthesized wavs
    dur_syn_total = sum(durs_syn)
    dur_ref_total = sum(durs_ref)

    # get the overall speed factor for speaking rate normalization
    speed = dur_syn_total / dur_ref_total
    print('average speed factor: {:.3f}'.format(speed))

    # get the duration of the entire audio file
    dur_total = librosa.get_duration(path=audio_file)

    # find the adjusted timestamps for the scaled segments   
    ts_ref = [() for _ in range(nsegments)]
    ts_allowed = [() for _ in range(nsegments)]
    ts_scaled = [() for _ in range(nsegments)]
    scaling_types = [0 for _ in range(nsegments)]
    previous_upper_bound = 0
    duration_offset = 0
    duration_offsets = [0 for _ in range(nsegments)]
    early_onset = 0.5
    gap = 0.1
    for i in range(nsegments):

        # get the timestamps about the reference chinese audio segment
        reference_audiofile = reference_audiofiles[i]
        parts = os.path.splitext(os.path.basename(reference_audiofile))[0].split('_')
        start_time_ref = round(float(parts[1]), 2)
        end_time_ref = round(float(parts[2]), 2)
        duration_ref = round(end_time_ref-start_time_ref, 2)
        ts_ref[i] = (start_time_ref, end_time_ref, duration_ref)

        # get the maximum-allowed timestamps based on the adjacent segments
        lower_bound0, upper_bound0 = find_bound(ts_lst, i, dur_total, gap=gap)
        # adjust the lowerbound to make sure it is larger than the previous upper bound
        lower_bound = max(previous_upper_bound, lower_bound0) # prevous upper bound already include the gap
        # only allow the scaled segment starts earlier than its reference segment by a small duration (e.g. early_onset=0.5 sec)
        lower_bound = max(lower_bound, start_time_ref-early_onset)
        duration_allowed = round(upper_bound0-lower_bound, 2)
        # case that the prevous upper bound has eaten up all the current segment duration
        if duration_allowed < 0:
            print(f'warning: {i}/{nsegments}: negative original allowed dur: ({lower_bound}, {upper_bound})')
            upper_bound = lower_bound + gap
            duration_allowed = gap
        else:
            upper_bound = upper_bound0
        ts_allowed[i] = (lower_bound, upper_bound, duration_allowed)
        # assert duration_allowed > 0, \
        #     f'{i}/{nsegments}: check lower and upper bound ({lower_bound}, {upper_bound})'

        # get the pre-adjusted duration of scaled segment (dur_scaled) based on dur_syn and overall speed factor
        duration_syn = durs_syn[i]
        duration_scaled = round(duration_syn/speed, 2)

        # case 1
        if duration_scaled <= duration_ref - duration_offset:
            # start_time_scaled = round(start_time_ref + duration_offset, 2)
            # end_time_scaled = round(start_time_scaled + duration_scaled, 2)
            start_time_scaled = round(start_time_ref + duration_offset, 2)
            # case 1.1: no speed change (i.e. speeds[i]==1)
            if duration_syn <= duration_ref - duration_offset:
                print(f'{i}/{nsegments}: case 1.1')
                scaling_types[i] = 1.1
                end_time_scaled = round(start_time_scaled + duration_syn, 2)
            # case 1.2: lower than avg speed change (i.e. speeds[i]<speed)
            else:
                print(f'{i}/{nsegments}: case 1.2')
                scaling_types[i] = 1.2
                end_time_scaled = end_time_ref
            duration_scaled = round(end_time_scaled - start_time_scaled, 2)
            ts_scaled[i] = (start_time_scaled, end_time_scaled, duration_scaled)
            duration_offset = 0

        # case 2    
        elif duration_scaled > duration_ref - duration_offset and duration_scaled <= duration_allowed:
            start_time_allowed = max(start_time_ref, lower_bound)
            # case 2.1 avg speed change, with extra alllowed duration, align to the beginning 
            if duration_scaled < upper_bound - start_time_allowed:
                print(f'{i}/{nsegments}: case 2.1')
                scaling_types[i] = 2.1
                start_time_scaled = round(start_time_allowed, 2)
                end_time_scaled = round(start_time_allowed + duration_scaled, 2)
            # case 2.2 avg speed change, without extra allowed duration, align to the middle
            else:
                print(f'{i}/{nsegments}: case 2.2')
                scaling_types[i] = 2.2
                start_time_scaled, end_time_scaled = get_scaled_ts(lower_bound, upper_bound, duration_scaled)
            # start_time_scaled = round(lower_bound, 2)
            # end_time_scaled = round(lower_bound+duration_scaled, 2)
            ts_scaled[i] = (start_time_scaled, end_time_scaled, duration_scaled)
            duration_offset = 0

        # case 3
        elif duration_scaled > duration_allowed:
            speed2 = np.ceil(duration_syn / duration_allowed * 100) / 100
            # case 3.1: above avg speed change, but under speed cap
            if speed2 <= speed_lim:
                print(f'{i}/{nsegments}: case 3.1')
                scaling_types[i] = 3.1
                duration_scaled = round(duration_syn / speed2, 2)
                start_time_scaled, end_time_scaled = get_scaled_ts(lower_bound, upper_bound, duration_scaled)
                # assert end_time_scaled - start_time_scaled == duration_scaled, \
                #     f'{i}/{nsegments}: scaled timestamp disqualified (case 2)'
                ts_scaled[i] = (start_time_scaled, end_time_scaled, duration_scaled)
                duration_offset = 0
            # case 3.2: over speed cap, need to have duration offset   
            else:
                print(f'{i}/{nsegments}: case 3.2')
                scaling_types[i] = 3.2
                duration_scaled = round(duration_syn / speed_lim, 2)
                start_time_scaled = round(lower_bound, 2)
                end_time_scaled = round(start_time_scaled + duration_scaled, 2)
                ts_scaled[i] = (start_time_scaled, end_time_scaled, duration_scaled)
                duration_offset = max(round(end_time_scaled - upper_bound, 2), 0)
                # raise Exception(f'{i}/{nsegments}: need to speed up to {speed2} to fit')
        else:
            raise Exception(f'{i}/{nsegments}: unexpected condition, please check!')
        duration_offsets[i] = duration_offset
        previous_upper_bound = round(end_time_scaled + gap, 2)

    # count scaling type freq.
    scaling_type_dct = {}
    for i in range(nsegments):
        k = scaling_types[i]
        if k in scaling_type_dct.keys():
            scaling_type_dct[k] += 1
        else:
            scaling_type_dct[k] = 1
    print('scaling type counts:')        
    for k in sorted(scaling_type_dct.keys()):
        percent = scaling_type_dct[k] / nsegments
        print(f'{k}: {scaling_type_dct[k]} ({percent*100:.2f}%)')

    # check duration offsets (cases that segment has to be shifted to a later start time due to no-space)
    duration_offsets_pos = [v for v in duration_offsets if v > 0]
    print(f'{len(duration_offsets_pos)}/{nsegments} segments has offset with avg. value ' + \
        f'{np.mean(duration_offsets_pos):.2f} secs.')

    # sanity check to ensure no overlaps in the adjacent segments
    for i in range(1, nsegments):
        start_time = ts_scaled[i][0]
        end_time_previous = ts_scaled[i-1][1]
        assert start_time >= end_time_previous, f'{i}/{nsegments}: {(end_time_previous-start_time):.2f} overlap'

    # write scaled segments and its statistics
    speeds = [0 for _ in range(nsegments)]
    durs_scaled = [0 for _ in range(nsegments)]
    durs_diff_abs = [0 for _ in range(nsegments)]
    durs_diff_percent = [0 for _ in range(nsegments)]
    xfactors = [0 for _ in range(nsegments)]
    offset_pairs = [(0,0) for _ in range(nsegments)]
    for i in range(nsegments):
        # collect statistics
        duration_syn = durs_syn[i]
        start_time_scaled, end_time_scaled, duration_scaled = ts_scaled[i]
        start_time_ref, end_time_ref, duration_ref = ts_ref[i]
        speeds[i] = duration_syn / duration_scaled # >1: dur_scaled < dur_syn, speed up syn segment
        xfactors[i] = duration_ref / duration_scaled # >1: dur_scaled < dur_ref, segment become shorter (ori -> scaled)
        offset_pairs[i] = (round(start_time_scaled-start_time_ref, 2), round(end_time_scaled-end_time_ref,2))
        durs_scaled[i] = duration_scaled
        durs_diff_abs[i] = round(duration_scaled-duration_ref, 2)
        durs_diff_percent[i] = durs_diff_abs[i] / duration_ref

        # print message
        msg1 = f'({i}/{nsegments}) ref ts: ({start_time_ref:.2f}, {end_time_ref:.2f}), ' + \
            f'scaled ts: ({start_time_scaled:.2f}, {end_time_scaled:.2f}), ' + \
            f'offset: ({offset_pairs[i][0]:.2f}, {offset_pairs[i][1]:.2f}) '
        msg2 = f'type: {scaling_types[i]}, ' + f'speed: {speeds[i]:.2f} (syn -> scaled), '
        msg3 = f'duration: {ts_ref[i][2]:.2f} (ori) -> {duration_scaled:.2f} (scaled), ' + \
            f'abs: {durs_diff_abs[i]:.2f}, pct: {durs_diff_percent[i]*100:.2f}%, '
        msg4 = f'xfactor: {xfactors[i]:.2f} (ori -> scaled)'
        print(msg1 + msg2 + msg3 + msg4)

        # write out scaled segment
        fid = f'{i:04d}_{start_time_scaled:.2f}_{end_time_scaled:.2f}_{speeds[i]:.2f}_{xfactors[i]:.2f}x'
        output_audiofile = os.path.join(out_dir, f'{fid}.wav')
        input_wav, output_wav, sr = adjust_speed(input_audiofiles[i], output_audiofile, speeds[i])

        # sanity check
        assert speeds[i] - (len(input_wav) / len(output_wav)) < 0.01, f'{i}/{nsegments} speed mis-match'

        # read meta from the input audio file
        jsonfilename = os.path.basename(input_audiofiles[i]).replace('.wav', '.json')
        jsonfile = os.path.join(meta_dir, jsonfilename)
        assert os.path.isfile(jsonfile), f'json file: {jsonfile} does not exist!'
        with open(jsonfile, encoding='utf-8') as f:
            meta0 = json.load(f)

        # save meta
        output_jsonfile = os.path.join(out_dir, f'{fid}.json')
        meta = {'fid': fid,
                'idx': i,
                'start-time-l1': round(ts_ref[i][0], 2),
                'end-time-l1': round(ts_ref[i][1], 2),
                'duration-l1': round(ts_ref[i][2], 2),
                'start-time-l2': round(ts_scaled[i][0], 2),
                'end-time-l2': round(ts_scaled[i][1], 2),
                'duration-l2': round(ts_scaled[i][2], 2),
                'scaling-type': scaling_types[i],
                'offset': offset_pairs[i],
                'speed': round(speeds[i], 2),
                'xfactor': round(xfactors[i], 2),
                'text-ori': meta0['text-original'],
                'text-tra': meta0['text-translated']}
        with open(output_jsonfile, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    # collect all output jsonfiles
    output_jsonfiles = sorted(glob.glob(os.path.join(out_dir, '*.json')))
    assert len(output_jsonfiles) == nsegments, '# of the output jsonfiles mis-match!'

    # get the header for the output meta csv
    with open(output_jsonfile) as f:
        meta = json.load(f)
    header = list(meta.keys())
    header = ['wav-file'] + header

    # construct rows
    rows = [() for _ in range(nsegments)]
    for i in range(nsegments):
        with open(output_jsonfiles[i]) as f:
            meta = json.load(f)
        output_audiofile = output_jsonfiles[i].replace('.json', '.wav')
        rows[i] = (output_audiofile, *meta.values())

    # write meta to csv
    out_csv_file = os.path.join(out_dir, f'{recording_id}_meta.csv')
    tuple2csv(rows, out_csv_file, delimiter='|', header=header, verbose=True)

    # check offset mean and std deviation
    offset_start = [offset_pairs[i][0] for i in range(nsegments)]
    offset_end = [offset_pairs[i][1] for i in range(nsegments)]
    offset_mean = [np.mean(offset_start), np.mean(offset_end)]
    offset_std = [np.std(offset_start), np.std(offset_end)]
    print(f'offset mean: [{offset_mean[0]:.2f}, {offset_mean[1]:.2f}]')
    print(f'offset std: [{offset_std[0]:.2f}, {offset_std[1]:.2f}]')