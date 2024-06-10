# generate Ukrainian synthesized wavs given transcript
# TTS model: fairseq mms-tts VITS
#
# example:
#  - ukr-transcription: /home/splola/kathol/SOFW/StaticVideos/scripts/MARCHE_AssessmentTacticalEnvironment.ukr.cor.sentids
#  - eng-trabscription: /home/splola/kathol/SOFW/StaticVideos/scripts/MARCHE_AssessmentTacticalEnvironment.eng.cor.sentids
#
# Zhenhao Ge, 2024-05-22

import os
from pathlib import Path
import torch
from transformers import VitsModel, AutoTokenizer
import librosa
import json
import time
import argparse
import numpy as np
import subprocess
import soundfile as sf

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

from ukrainian_tts.formatter import preprocess_text
from ukrainian_tts.stress import sentence_to_stress, stress_dict, stress_with_model
from sofw.utils import read_trans, parse_fid

def get_hostname():
    hostname = subprocess.check_output('hostname').decode('ascii').rstrip()
    return hostname

def stress_text(text_preprocessed, stress):
    if stress == 'model':
        stress = True
    else:
        stress = False
    text_stressed = sentence_to_stress(
        text_preprocessed, stress_with_model if stress else stress_dict)
    return text_stressed  

def synthesize(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    wav = np.ravel(np.array(output)) 

    return wav

def parse_args():
    usage = 'usage: generate ukr wavs given transcript with fairseq mms-tts model'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--trans-file', type=str, help='input transcription file')
    parser.add_argument('--model-path', type=str, help='tts model path')
    parser.add_argument('--output-path', type=str, help='output path')
    parser.add_argument('--stress', choices=['dictionary', 'model'], help='stress option')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # work_path = os.getcwd() # e.g., '/home/users/zge/code/repo/ukr-tts'
    # trans_path = '/home/splola/kathol/SOFW/StaticVideos/scripts'
    # recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    # args.trans_file = os.path.join(trans_path, '{}.ukr.cor.sentids'.format(recording_id))
    # args.model_path = os.path.join(work_path, 'model', 'mms-tts')
    # args.stress = 'dictionary'
    # args.output_path = os.path.join(work_path, 'outputs', 'sofw', 'mms-tts', recording_id, args.stress)

    print('transcription file: {}'.format(args.trans_file))
    print('model path: {}'.format(args.model_path))
    print('output path: {}'.format(args.output_path))
    print('stress: {}'.format(args.stress))

    hostname = get_hostname()
    print('hostname: {}'.format(hostname))

    # create the output path if it does not exist
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
        print('created output dir: {}'.format(args.output_path))
    else:
        print('using existing output dir: {}'.format(args.output_path))    
    
    # load TTS model
    model = VitsModel.from_pretrained(args.model_path)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # get the list of sentences (id, text) from the transcription file
    sentences = read_trans(args.trans_file)
    num_sents = len(sentences)
    print('there are {} sentences in transcription file {}'.format(num_sents, args.trans_file))

    for i in range(num_sents):

        print('processing sentence {}/{} ...'.format(i+1, num_sents))

        # get fid and text from sentence tuple
        fid, text = sentences[i]

        text_preprocessed = preprocess_text(text)
        text_accented = stress_text(text_preprocessed, args.stress)

        # print('text original: {}'.format(text))
        # print('text preprocessed: {}'.format(text_preprocessed))
        # print('text stressed: {}'.format(text_stressed))
        
        # synthesis
        start = time.time()
        wav = synthesize(text_accented, model, tokenizer)
        end = time.time()
        dur_proc = end - start
        dur_syn = len(wav) / model.config.sampling_rate
        rtf = dur_proc / dur_syn

        # save the synthesized wav
        output_file = os.path.join(args.output_path, '{}.wav'.format(fid))
        sf.write(output_file, wav, model.config.sampling_rate)

        # parse the fid to get the timed duration
        fid_seq, start_time, end_time = parse_fid(fid)
        dur_timed = end_time - start_time
        timed2syn = dur_timed / dur_syn

        # collect the meta info
        meta = {'syn-wav': output_file,
                'text-original': text,
                'text-preprocessed': text_preprocessed,
                'text-accented': text_accented,
                'model': args.model_path,
                'stress': args.stress,
                'rtf': rtf,
                'dur-proc': dur_proc,
                'dur-syn': dur_syn,
                'fid-seq': fid_seq,
                'start-time': start_time,
                'end-time': end_time,
                'dur-timed': dur_timed,
                'timed2syn': timed2syn}

        # save the meta info as a json file        
        output_jsonfile = output_file.replace('.wav', '.json')
        with open(output_jsonfile, 'w') as fp:
            json.dump(meta, fp, indent=2, ensure_ascii=False)

    # sound = AudioSegment.from_file(output_file, 'wav')
    # speed = 1.2
    # change_speed_only(sound, speed)

