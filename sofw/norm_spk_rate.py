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

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

from sofw.utils import get_dur_from_meta
from sofw.utils import dl2csv
from audio import adjust_speed

def parse_args():
    usage = 'usage: normalize speaking rate'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--input-path', type=str, help='input path')
    parser.add_argument('--output-path', type=str, help='output path')
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
    # args.input_path = os.path.join(work_path, 'outputs', 'sofw', 'espnet',
    #     recording_id, '{}-{}'.format(voice, stress))
    # args.output_path = args.input_path + '_scaled'

    # sanity check
    assert os.path.isdir(args.input_path), \
        'input path: {} does not exist!'.format(args.input_path)

    # create output path
    if os.path.isdir(args.output_path):
        print('using existing output path: {}'.format(args.output_path))
    else:
        os.makedirs(args.output_path)
        print('created output path: {}'.format(args.output_path))        

    print('input path: {}'.format(args.input_path))
    print('output path: {}'.format(args.output_path))

    # get the list of wav and json files 
    input_wavfiles = sorted(glob.glob(os.path.join(args.input_path, '*.wav')))
    input_jsonfiles = sorted(glob.glob(os.path.join(args.input_path, '*.json')))
    num_wavfiles = len(input_wavfiles)
    num_jsonfiles = len(input_jsonfiles)
    assert num_wavfiles == num_jsonfiles, '#wavfiles and #jsonfiles mismatch!'
    num_sentences = num_wavfiles
    del num_wavfiles, num_jsonfiles

    # get durations for the referenced and synthesized wavs
    durs_ref, durs_syn = get_dur_from_meta(input_jsonfiles)
    dur_ref_total = sum(durs_ref)
    dur_syn_total = sum(durs_syn)

    # get the overall speed factor for speaking rate normalization
    speed = dur_syn_total / dur_ref_total
    print('speed factor: {:.3f}'.format(speed))

    # time-scaling to normalize the speaking rate
    if speed > 1:
        speeds = [0 for _ in range(num_sentences)]
        for i in range(num_sentences):

            # generate the time-scaled wav
            input_wavfile = input_wavfiles[i]
            wavname = os.path.basename(input_wavfile)
            output_wavfile = os.path.join(args.output_path, wavname)
            input_wav, output_wav, sr = adjust_speed(input_wavfile, output_wavfile, speed)

            input_nsamples = len(input_wav)
            output_nsamples = len(output_wav)
            speeds[i] = input_nsamples / output_nsamples

            # get the scaled duration
            dur_scaled = output_nsamples / sr

            # generate meta json file for the time-scaled wav
            input_jsonfile = input_jsonfiles[i]
            with open(input_jsonfile, 'r') as f:
                meta = json.load(f)
            meta['dur-scaled'] = dur_scaled
            output_jsonfile = output_wavfile.replace('.wav', '.json')
            with open(output_jsonfile, 'w') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        speed2 = np.mean(speeds)    
        print('time-scaled with speed={:.2f} in output path: {}'.format(
            speed2, args.output_path))
    else:
        print('speed={}, no time scaling needed')

    # save the duration info as csv
    if speed > 1:
        dct_lst = [{} for _ in range(num_sentences)]
        for i in range(num_sentences):

            input_jsonfile = input_jsonfiles[i]
            jsonname = os.path.basename(input_jsonfile)
            output_jsonfile = os.path.join(args.output_path, jsonname)
            assert os.path.isfile(output_jsonfile), \
                'output jsonfile: {} does not exist!'.format(output_jsonfile)

            with open(output_jsonfile, 'r') as f:
                meta = json.load(f)
            dct_lst[i] = {'fid-seq': meta['fid-seq'],
                          'start-time': meta['start-time'],
                          'end-time': meta['end-time'],
                          'dur-ref': meta['dur-timed'],
                          'dur-syn': meta['dur-syn'],
                          'dur-scaled': meta['dur-scaled']}

        # save the dict list as csv file
        output_csvfile = os.path.join(args.output_path, 'dur_stats.csv')
        header = list(dct_lst[0].keys())
        dl2csv(dct_lst, header, output_csvfile, verbose=True)




     



            