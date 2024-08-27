# normalize speaking rate to match the total (average) synthesized output duration, 
# with the total (average) original reference duration
#
# example:
#   - synthesized output: outputs/sofw/espnet/{recording-id}/dmytro-dictionary
#   - scaled (speaking rate, or duration normalized) output: outputs/sofw/espnet/{recording-id}/dmytro-dictionary_scaled 
#
# Zhenhao Ge, 2024-05-23

import os
from pathlib import Path
import glob
import json
import numpy as np
import argparse
import librosa

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

from sofw.utils import get_dur_from_meta, get_dur_from_file
from sofw.utils import dl2csv, set_path, empty_dir
from audio import adjust_speed

keywords = '.16000,_converted,_ukr,_resampled,_paired,_unpaired'
keywords = keywords.split(',')

def filter_path(paths, keywords):
    for kw in keywords:
        paths = [f for f in paths if kw not in f]
    return paths

def parse_args():
    usage = 'usage: normalize speaking rate'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--input-path', type=str, help='input path of tts speech in its original speed')
    parser.add_argument('--output-path', type=str, help='output path of the output time-scaled tts speech')
    parser.add_argument('--reference-path', type=str, help='reference of the real speech ' + \
        '(get actual duration to match in case there is no meta json file)')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # work_path = os.getcwd()
    # recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    # voice = 'dmytro'
    # stress = 'dictionary'
    # args = argparse.ArgumentParser()
    # # args.input_path = os.path.join(work_path, 'outputs', 'sofw', 'espnet',
    # #     recording_id, '{}-{}'.format(voice, stress))
    # args.input_path = os.path.abspath(os.path.join(work_path, os.pardir, 'free-vc', 'outputs',
    #     recording_id, f'freevc-24_{voice}-{stress}'))
    # args.output_path = args.input_path + '_scaled'
    # args.reference_path = os.path.join(work_path, 'data', recording_id, 'segments')

    # sanity check
    assert os.path.isdir(args.input_path), \
        'input path: {} does not exist!'.format(args.input_path)
    assert os.path.isdir(args.reference_path), \
        'reference path: {} does not exist!'.format(args.reference_path)

    # create output path
    set_path(args.output_path, verbose=True)
    empty_dir(args.output_path)

    print('input path: {}'.format(args.input_path))
    print('output path: {}'.format(args.output_path))
    print('reference path: {}'.format(args.reference_path))

    # get the list of wav and json files 
    input_wavfiles = sorted(glob.glob(os.path.join(args.input_path, '*.wav')))
    input_wavfiles = filter_path(input_wavfiles, keywords)
    input_jsonfiles = sorted(glob.glob(os.path.join(args.input_path, '*.json')))
    num_wavfiles = len(input_wavfiles)
    num_jsonfiles = len(input_jsonfiles)

    if num_jsonfiles > 0:
        assert num_wavfiles == num_jsonfiles, '#wavfiles and #jsonfiles mismatch!'
        num_sentences = num_wavfiles
        del num_wavfiles, num_jsonfiles

        # get durations for the referenced and synthesized wavs from meta files
        durs_ref, durs_syn = get_dur_from_meta(input_jsonfiles)

        has_meta = True

    else:

        ref_wavfiles = sorted(glob.glob(os.path.join(args.reference_path, '*.wav')))
        ref_wavfiles = filter_path(ref_wavfiles, keywords)
        num_ref_wavfiles = len(ref_wavfiles)
        assert num_ref_wavfiles == num_wavfiles, "# input and reference wavfiles mismatch!"
        num_sentences = num_wavfiles
        del num_wavfiles, num_ref_wavfiles

        # get durations for the referenced and synthesized wavs from files directly
        durs_ref = get_dur_from_file(ref_wavfiles)
        durs_syn = get_dur_from_file(input_wavfiles)

        has_meta = False

    print(f'has meta: {has_meta}')

    # get the total duration for the referenced and synthesized wavs
    dur_ref_total = sum(durs_ref)
    dur_syn_total = sum(durs_syn)

    # get the overall speed factor for speaking rate normalization
    # (1.262 for MARCHE_AssessmentTacticalEnvironment)
    speed = dur_syn_total / dur_ref_total
    print('speed factor: {:.3f}'.format(speed))

    assert speed > 1, 'speed={}, no time scaling needed, stop here!'

    # time-scaling to normalize the speaking rate
    speeds = [0 for _ in range(num_sentences)] # sentence-wise dur_syn/dur_ref
    durs_scaled = [0 for _ in range(num_sentences)]
    durs_diff_abs = [0 for _ in range(num_sentences)] # sentence-wise (dur_scaled - dur_ref)
    durs_diff_percent = [0 for _ in range(num_sentences)] # sentence-wise (dur_scaled-dur_ref)/dur_ref
    xfactors = [0 for _ in range(num_sentences)] # sentence-wise dur_ref/dur_scaled
    for i in range(num_sentences):

        # generate the time-scaled wav
        input_wavfile = input_wavfiles[i]
        wavname = os.path.basename(input_wavfile)
        output_wavfile = os.path.join(args.output_path, wavname)
        input_wav, output_wav, sr = adjust_speed(input_wavfile, output_wavfile, speed)

        input_nsamples = len(input_wav)
        output_nsamples = len(output_wav)
        speeds[i] = input_nsamples / output_nsamples # should ber always close to speed

        # get the scaled duration
        durs_scaled[i] = output_nsamples / sr

        # get duration change stats
        durs_diff_abs[i] = durs_scaled[i] - durs_ref[i] # near 0, positive: longer, negative: shorter
        durs_diff_percent[i] = durs_diff_abs[i] / durs_ref[i] # near 0%, positive: longer, negative: shorter

        # get speed factor (1X: normal, 1X+: faster, 1X-: slower)
        xfactors[i] = durs_ref[i] / durs_scaled[i]

        # generate meta json file for the time-scaled wav
        if has_meta:
            input_jsonfile = input_jsonfiles[i]
            with open(input_jsonfile, 'r') as f:
                meta = json.load(f)
            meta['dur-scaled'] = durs_scaled[i]
            output_jsonfile = output_wavfile.replace('.wav', '.json')
            with open(output_jsonfile, 'w') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

    speed2 = np.mean(speeds) # should be very close to speed
    print('time-scaled with speed={:.3f} in output path: {}'.format(
        speed2, args.output_path))
    assert np.abs(speed2 - speed) < 0.001, 'speed2 should be very close to speed!'

    # check if there is overlaps
    # gap[i] stores the gap dur between segment[i] and segment [i+1] (pos: with gap, neg, with overlap)
    gaps = np.zeros(num_sentences-1)
    for i in range(num_sentences-1):

        # get the current end time
        wavname_current = os.path.basename(input_wavfiles[i])
        parts = os.path.splitext(wavname_current)[0].split('_')
        start_time_current = float(parts[1])
        output_wavfile_current = os.path.join(args.output_path, wavname_current)
        duration_current = librosa.get_duration(path=output_wavfile_current)
        end_time_current = start_time_current + duration_current

        # get the next start time
        wavname_next = os.path.basename(input_wavfiles[i+1])
        parts = os.path.splitext(wavname_next)[0].split('_')
        start_time_next = float(parts[1])

        gaps[i] = start_time_next - end_time_current
    print(f'average gap: {np.mean(gaps):.2f} seconds')

    # check if there are overlaps between adjacent segments
    # overlap is -gap, but it only counts the positive ones (i.e., when gap<0)
    overlaps = []
    for i, gap in enumerate(gaps):
        if gap < 0:
            print(f'{i}/{num_sentences}: overlap {-gap:.2f} seconds!')
            overlaps.append(-gap)
    print(f'{len(overlaps)}/{num_sentences-1} segment pairs have overlaps with average {np.mean(overlaps):.2f} seconds!')

    # save the duration info as csv
    dct_lst = [{} for _ in range(num_sentences)]
    for i in range(num_sentences):

        if has_meta:

            input_jsonfile = input_jsonfiles[i]
            jsonname = os.path.basename(input_jsonfile)
            output_jsonfile = os.path.join(args.output_path, jsonname)
            assert os.path.isfile(output_jsonfile), \
                'output jsonfile: {} does not exist!'.format(output_jsonfile)

            with open(output_jsonfile, 'r') as f:
                meta = json.load(f)
            dct_lst[i] = {'fid-seq': meta['fid-seq'],
                        'start-time': round(meta['start-time'],2),
                        'end-time': round(meta['end-time'],2),
                        'dur-ref': round(meta['dur-timed'],2),
                        'dur-syn': round(meta['dur-syn'],2),
                        'dur-scaled': round(meta['dur-scaled'],2),
                        'dur-diff-abs': round(durs_diff_abs[i],2),
                        'dur-diff-percent': round(durs_diff_percent[i],2),
                        'xfactor': round(xfactors[i],2)}

        else:

            wavname = os.path.basename(input_wavfiles[i])
            # jsonname = wavname.replace('.wav', '.json')
            # output_jsonfile = os.path.join(args.output_path, jsonname)

            parts = os.path.splitext(wavname)[0].split('_')
            fid_seq = parts[0]
            start_time = round(float(parts[1]), 2)
            end_time = round(float(parts[2]), 2)
            dct_lst[i] = {'fid-seq': fid_seq,
                            'start-time': start_time,
                            'end-time': end_time,
                            'dur-ref': round(durs_ref[i],2),
                            'dur-syn': round(durs_syn[i],2),
                            'dur-scaled': round(durs_scaled[i],2),
                            'dur-diff-abs': round(durs_diff_abs[i],2),
                            'dur-diff-percent': round(durs_diff_percent[i],2),
                            'xfactor': round(xfactors[i],2)}

    # save the dict list as csv file
    output_csvfile = os.path.join(args.output_path, 'dur_stats.csv')
    header = list(dct_lst[0].keys())
    dl2csv(dct_lst, header, output_csvfile, verbose=True)

    # check the total durations in the reference and scaled audio files
    durs_ref = [dct['dur-ref'] for dct in dct_lst]
    durs_scaled = [dct['dur-scaled'] for dct in dct_lst]
    print(f'total duration from the reference audio files: {sum(durs_ref)} seconds')
    print(f'total duration from the scaled audio files: {sum(durs_scaled)} seconds')