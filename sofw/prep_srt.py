# prepare the subtitle file in the srt format
# 
# Zhenhao Ge, 2024-07-16

import os, sys
from pathlib import Path
import argparse

# set paths
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

def sec2hms(seconds, precision=2):
    """convert seconds to hh:mm:ss,ms, where ms is 3 digits, e.g., 01:03:2,180"""
    hh = f'{int(seconds/3600):02d}' # 2 digits (00 ~ 99)
    mm = f'{int((seconds%3600)/60):02d}' # 2 digits (0 ~ 59)
    secs = seconds%3600%60 
    ss = f'{int(secs):d}' # either 1 or 2 digits (0 ~ 59)
    ms =  f'{int(round(secs-float(ss), precision)*1000):03d}' # 3 digits (000 ~ 999)
    hms = f'{hh}:{mm}:{ss},{ms}'
    return hms

def parse_args():
    usage = 'usage: prepare the subtitle file in the srt format'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--meta-file', type=str, help='meta file containing timestamps and texts')
    parser.add_argument('--srt-file', type=str, help='subtitle file in srt format')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()
    # recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    # voice = 'dmytro'
    # stress = 'dictionary'
    # meta_dir = os.path.join(home_dir, 'code', 'repo', 'free-vc', 'outputs', recording_id,
    #     f'freevc-24_{voice}-{stress}_scaled')
    # args.meta_file = os.path.join(meta_dir, f'{recording_id}_meta.csv')
    # args.srt_file = os.path.join(meta_dir, f'{recording_id}_ukr.srt')

    # check file/dir existence
    assert os.path.isfile(args.meta_file), f'meta file: {args.meta_file} does not exist!'

    # localize arguments
    meta_file = args.meta_file
    srt_file = args.srt_file

    # print arguments
    print(f'meta file: {meta_file}')
    print(f'srt file: {srt_file}')

    # read meta file
    lines = open(meta_file, 'r').readlines()
    header = lines[0].strip().split('|')
    lines = lines[1:]
    nsegments = len(lines)
    print(f'# of segments: {nsegments}')

    # get the index of start time, end time, and text
    colname_lst = ['start-time-l2', 'end-time-l2', 'text-tra']
    idx_dct = {k:header.index(k) for k in colname_lst}

    # find tuple list of (start time, end time, text)
    tuple_lst = [() for _ in range(nsegments)]
    for i in range(nsegments):
        parts = lines[i].strip().split('|')
        start_time_l2 = round(float(parts[idx_dct['start-time-l2']]), 2)
        end_time_l2 = round(float(parts[idx_dct['end-time-l2']]), 2)
        text_tra = parts[idx_dct['text-tra']]
        tuple_lst[i] = (start_time_l2, end_time_l2, text_tra)

    # construct rows in the srt file
    rows = []
    for i in range(nsegments):
        rows.append(str(i))
        start_time_l2, end_time_l2, text_tra = tuple_lst[i]
        start_hms = sec2hms(start_time_l2)
        end_hms = sec2hms(end_time_l2)
        rows.append(f'{start_hms} --> {end_hms}')
        rows.append(text_tra)
        rows.append('')

    # write out the srt file
    with open(srt_file, 'w') as f:
        f.writelines('\n'.join(rows) + '\n')
    print(f'wrote the srt file: {srt_file}')     