import os
from pathlib import Path
import argparse

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

from audio import audioread, audiowrite
from sofw.utils import empty_dir

def parse_srt(srt_file):
    fid2text = {}
    lines = open(srt_file, 'r').readlines()
    for i, line in enumerate(lines):
        parts = line.strip().split()
        fid = parts[0]
        text = ' '.join(parts[1:])
        fid2text[fid] = text
    return fid2text

def write_scp(scp_file, fids, wavpaths):

    with open(scp_file, 'w') as f:
        for fid, wavpath in zip(fids, wavpaths):
            f.write('{} {}\n'.format(fid, wavpath))
    print('wrote wav.scp: {}'.format(scp_file))

def write_text(text_file, fids, texts):

    with open(text_file, 'w') as f:
        for fid, text in zip(fids, texts):
            f.write('{} {}\n'.format(fid, text))
    print('wrote text: {}'.format(text_file))

def write_utt2spk(utt2spk_file, fids, spk_id):

    with open(utt2spk_file, 'w') as f:
        for fid in fids:
            f.write('{} {}\n'.format(fid, spk_id))
    print('wrote utt2spk: {}'.format(utt2spk_file))             

def parse_args():
    usage = 'usage: extract audio segments'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--wav-file', type=str,
        help='single audio wav file to extract segments from')
    parser.add_argument('--srt-file', type=str,
        help='srt file contains id and text')
    parser.add_argument('--out-path', type=str,
        help='output path to save segments')
    return parser.parser_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # work_path = os.getcwd()
    # sofw_path = os.path.join(home_path, 'code', 'unclassified', 'SOFW')
    # recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    # args.wav_file = os.path.join(sofw_path, 'kathol', 'StaticVideos', 'data', 'audio',
    #     '{}.wav'.format(recording_id))
    # # args.srt_file = os.path.join(sofw_path, 'kathol', 'StaticVideos', 'scripts',
    # #     '{}.eng.sentids'.format(recording_id)) # old version
    # args.srt_file = os.path.join(sofw_path, 'kathol', 'StaticVideos', 'data', 'corrections',
    #     '{}-ASRcorrected1.v1.eng.sentids'.format(recording_id)) # new version
    # args.out_path = os.path.join(work_path, 'data', recording_id, 'segments')

    # check file existance
    assert os.path.isfile(args.wav_file), 'wav file: {} does not exist!'.format(args.wav_file)
    assert os.path.isfile(args.srt_file), 'srt file: {} does not exist!'.format(args.srt_file)

    if os.path.isdir(args.out_path):
        empty_dir(args.out_path)
        print('use existing output path: {}'.format(args.out_path))
    else:
        os.makedirs(args.out_path)
        print('created output path: {}'.format(args.out_path))

    # localize arguments
    wav_file = args.wav_file
    srt_file = args.srt_file
    out_path = args.out_path

    # extract the fid2text dict from the srt file
    fid2text = parse_srt(srt_file)
    fids = sorted(fid2text.keys())

    # get the number of fids
    num_fids = len(fids)
    print('# of fids: {}'.format(num_fids))

    # read audio segments from wav file and write them to out path 
    out_files = ['' for _ in range(num_fids)]
    for i, fid in enumerate(fids):

        fid = fids[i]
        start_time = float(fid.split('_')[1])
        end_time = float(fid.split('_')[2])
        duration = end_time - start_time

        # read audio segment
        data, params = audioread(wav_file, start_time, duration)

        # write audio segment
        out_file = os.path.join(out_path, '{}.wav'.format(fid))
        audiowrite(out_file, data, params)
        print('wrote {}'.format(out_file))

        out_files[i] = out_file

    # kaldi data prep (wav.scp, text)
    kaldi_data_path = os.path.dirname(out_path)
    os.makedirs(kaldi_data_path, exist_ok=True)

    # write wav.scp
    scp_file = os.path.join(kaldi_data_path, 'wav.scp')
    write_scp(scp_file, fids, out_files)

    # write text
    texts = [fid2text[fid] for fid in fids]
    text_file = os.path.join(kaldi_data_path, 'text')
    write_text(text_file, fids, texts)

    # write utt2spk
    spk_id = '0'
    utt2spk_file = os.path.join(kaldi_data_path, 'utt2spk')
    write_utt2spk(utt2spk_file, fids, spk_id)
