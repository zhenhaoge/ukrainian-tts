# copy multiple versions of audio segments to one place for demo
#
# version 1: original english audio segments
# version 2: synthesized ukrainian audio segments
# version 3: voice-converted ukrainian audio segments
# version 4: voice-converted and time-scaled ukrainian audio segments
#
# Zhenhao Ge, 2024-06

import os
from pathlib import Path
import glob
import argparse
import shutil

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

from sofw.utils import set_path

def filter_path(paths, keywords):
    for kw in keywords:
        paths = [f for f in paths if kw not in f]
    return paths

def copy_files(in_path, out_path, ext, keywords, appendix, verbose=True):
    data_files = glob.glob(os.path.join(in_path, f'*.{ext}'))
    data_files = filter_path(data_files, keywords)
    num_data_files = len(data_files)
    for f in data_files:
        fn2 = os.path.splitext(os.path.basename(f))[0] + appendix + f'.{ext}'
        f2 = os.path.join(out_path, fn2)
        shutil.copyfile(f, f2)
    if verbose:    
        print(f'copied {num_data_files} files from {in_path} to {out_path} with appendix {appendix}')    

def parse_args():
    usage = 'usage: copy multiple versions of audio segments to one place for demo'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--ori-path', type=str, help='path to the original english audio segments')
    parser.add_argument('--syn-path', type=str, help='path to the synthesized ukrainian audio segments')
    parser.add_argument('--converted-path', type=str, help='path to the converted ukrainian audio segments')
    parser.add_argument('--scaled-path', type=str, help='path to the time-scaled ukrainian audio segments')
    parser.add_argument('--out-path', type=str, help='output path')
    parser.add_argument('--keywords', type=str, \
        help="seperated by comma to filter out, e.g., '16000', '_new'")

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # interactive mode
    args = argparse.ArgumentParser()

    recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    voice = 'dmytro'
    stress = 'dictionary'
    spk_folder = f'{voice}-{stress}'

    args.ori_path = os.path.join(work_path, 'data', recording_id, 'segments')
    args.syn_path = os.path.join(work_path, 'outputs', 'sofw', 'espnet', recording_id, spk_folder)
    args.converted_path = os.path.join(home_path, 'code', 'repo', 'free-vc', 'outputs', recording_id, f'freevc-24_{spk_folder}')
    args.scaled_path = args.converted_path + '_scaled'
    args.out_path = os.path.join(work_path, 'outputs', 'sofw', 'demo', recording_id, spk_folder)
    args.keywords = '.16000,_new,_converted,_paired,_unpaired, _v1, _v2, _v2, _v4'

    # localize arguments
    ori_path = args.ori_path
    syn_path = args.syn_path
    converted_path = args.converted_path
    scaled_path = args.scaled_path
    out_path = args.out_path
    keywords = args.keywords.split(',')

    # check path existence
    assert os.path.isdir(ori_path), f'path to the original english audio segments: {ori_path} does not exist!'
    assert os.path.isdir(syn_path), f'path to the synthesized ukrainian audio segments: {syn_path} does not exist!'
    assert os.path.isdir(converted_path), f'path to the voice-converted ukrainian audio segments: {converted_path} does not exist!'
    assert os.path.isdir(scaled_path), f'path to the scaled english audio segments: {scaled_path} does not exist!'

    # set the output path
    set_path(out_path, verbose=True)

    # print paths
    print(f'path to the original english audio segments: {ori_path}')
    print(f'path to the synthesized ukrainian audio segments: {syn_path}')
    print(f'path to the voice-converted ukrainian audio segments: {converted_path}')
    print(f'path to the voice-converted and time-scaled ukrainian audio segments: {scaled_path}')

    # copy files of multiple version to the same output folders with renaming with appendix
    ext = 'wav'
    copy_files(ori_path, out_path, ext, keywords, appendix='_v1.eng', verbose=True)
    copy_files(syn_path, out_path, ext, keywords, appendix='_v2.ukr', verbose=True)
    copy_files(converted_path, out_path, ext, keywords, appendix='_v3.converted', verbose=True)
    copy_files(scaled_path, out_path, ext, keywords, appendix='_v4.converted-scaled', verbose=True)
