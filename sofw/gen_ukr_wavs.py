# generate Ukrainian synthesized wavs given transcript
# TTS model: espnet conformer-based FastSpeech2 + HiFi-GAN
#
# example:
#  - ukr-transcription: /home/splola/kathol/SOFW/StaticVideos/scripts/MARCHE_AssessmentTacticalEnvironment.ukr.cor.sentids
#  - eng-trabscription: /home/splola/kathol/SOFW/StaticVideos/scripts/MARCHE_AssessmentTacticalEnvironment.eng.cor.sentids
#
# Zhenhao Ge, 2024-05-22

import os
from pathlib import Path
import librosa
import json
import argparse

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

from ukrainian_tts.tts import TTS, Voices, Stress
from sofw.utils import read_trans, parse_fid, empty_dir

def parse_args():
    usage = 'usage: generate ukr wavs given transcript with espnet model'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--trans-file1', type=str, help='transcription file for the source language')
    parser.add_argument('--trans-file2', type=str, help='transcription file for the target language')
    parser.add_argument('--model-path', type=str, help='tts model path')
    parser.add_argument('--output-path', type=str, help='output path')
    parser.add_argument('--voice', choices=['tetiana', 'mykyta', 'lada', 'dmytro', 'oleksa'],
        help='voice options')
    parser.add_argument('--stress', choices=['dictionary', 'model'], help='stress options')
    parser.add_argument('--device', type=str, default='cpu', help='gpu/cpu device')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # recording_id = 'MARCHE_AssessmentTacticalEnvironment'

    # # old version
    # trans_path = '/home/splola/kathol/SOFW/StaticVideos/scripts'
    # args.trans_file1 = os.path.join(trans_path, '{}.eng.cor.sentids'.format(recording_id))
    # args.trans_file2 = os.path.join(trans_path, '{}.ukr.cor.sentids'.format(recording_id))

    # # updated version
    # trans_path = '/home/splola/kathol/SOFW/StaticVideos/data/corrections'
    # args.trans_file1 = os.path.join(trans_path, f'{recording_id}-ASRcorrected1.v1.eng.sentids')
    # args.trans_file2 = os.path.join(trans_path, f'{recording_id}-ASRcorrected1.v1.ukr.sentids')

    # args.model_path = os.path.join(work_path, 'model', 'espnet')
    # args.voice = 'dmytro' # options: tetiana, mykyta, lada, dmytro, oleksa
    # args.stress = 'dictionary' # options: dictionary, model
    # output_folder = '{}-{}'.format(args.voice, args.stress)
    # args.output_path = os.path.join(work_path, 'outputs', 'sofw', 'espnet', recording_id, output_folder)
    # args.device = 'cuda:0' # 'cuda', 'cuda:x', or 'cpu'

    # sanity check
    assert os.path.isfile(args.trans_file1), \
        'transcription file for the target language: f{args.trans_file1} does not exist!'
    assert os.path.isfile(args.trans_file2), \
        'transcription file for the target language: f{args.trans_file2} does not exist!'

    print('transcription file 1 (source language): {}'.format(args.trans_file1))
    print('transcription file 2 (target language): {}'.format(args.trans_file2))
    print('model path: {}'.format(args.model_path))
    print('voice: {}'.format(args.voice))
    print('stress: {}'.format(args.stress))
    print('output path: {}'.format(args.output_path))
    print('device: {}'.format(args.device))

    # create the output path if it does not exist
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
        print('created output dir: {}'.format(args.output_path))
    else:
        # empty_dir(args.output_path)
        print('using existing output dir: {}'.format(args.output_path))

    # load TTS model
    tts = TTS(device=args.device, cache_folder=args.model_path)

    # get the list of sentences (id, text) from the transcription file 1
    sentences1 = read_trans(args.trans_file1)
    num_sents1 = len(sentences1)
    fids1 = [fid for (fid, text) in sentences1]

    # get the list of sentences (id, text) from the transcription file 2
    sentences2 = read_trans(args.trans_file2)
    num_sents2 = len(sentences2)
    fids2 = [fid for (fid, text) in sentences2]

    assert num_sents1 == num_sents2, '#sentences in the source and target transcription files mis-match!'
    open(os.path.join(work_path, 'fid1.txt'), 'w').writelines('\n'.join(fids1))
    open(os.path.join(work_path, 'fid2.txt'), 'w').writelines('\n'.join(fids2))
    fid_diff_pairs = [(fid1, fid2) for (fid1, fid2) in zip(fids1, fids2) if fid1!=fid2]
    if len(fid_diff_pairs) > 0:
        print('fids in the source and target transcription files mis-match, take the source timestamps!')
    num_sents = num_sents1
    fids = fids1
    del num_sents1, num_sents2, fids1, fids2
    print('there are {} sentences in source trans file {} and target trans file {}'.format(
        num_sents, os.path.basename(args.trans_file1), os.path.basename(args.trans_file2)))

    for i in range(num_sents):

        print('processing sentence {}/{} ...'.format(i+1, num_sents))

        # get the text from the source sentence
        _, text0 = sentences1[i]

        # get the fid and the text from the target sentence
        fid, text = sentences2[i]

        # get output file path
        output_file = os.path.join(args.output_path, '{}.wav'.format(fid))

        # synthesis
        with open(output_file, mode="wb") as file:
            output_fp, output_text, rtf = tts.tts(text, args.voice, args.stress, file)

        # get the output duration
        output_wav, _ = librosa.load(output_file, sr=tts.synthesizer.fs)
        dur_syn = len(output_wav) / tts.synthesizer.fs
        dur_proc = dur_syn * rtf

        # parse the fid to get the timed duration
        fid_seq, start_time, end_time = parse_fid(fid)
        duration_timed = end_time - start_time

        # get the ratio of input wav duration vs output (syn) wav duration
        # (if <1, means the duration of the output wav is longer)
        timed2syn = duration_timed / dur_syn

        # collect the meta info
        meta = {'syn-wav': output_file,
                'text-original': text0,
                'text-translated': text,
                'text-accented': output_text,
                'model': args.model_path,
                'voice': args.voice,
                'stress': args.stress,
                'rtf': rtf,
                'dur-proc': dur_proc,
                'dur-syn': dur_syn,
                'fid-seq': fid_seq,
                'start-time': start_time,
                'end-time': end_time,
                'dur-timed': duration_timed,
                'timed2syn': timed2syn}

        # save the meta info as a json file        
        output_jsonfile = output_file.replace('.wav', '.json')
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2, ensure_ascii=False)