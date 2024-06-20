# Convert source audio segments to target audio segments using SpeechT5-VC
# 
# Zhenhao Ge, 2024-06-11 

import os
from pathlib import Path
import argparse
import librosa
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier
import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf
from kaldiio import load_ark
import numpy as np
import glob

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

# set the samlpe rate of the data used to train the speaker embedding model
sample_rate = 16000

spk_model = {
    "speechbrain/spkrec-xvect-voxceleb": 512, 
    "speechbrain/spkrec-ecapa-voxceleb": 192,
}

def filter_path(paths, keywords):
    for kw in keywords:
        paths = [f for f in paths if kw not in f]
    return paths

def f2embed(wav_file, classifier, size_embed, sample_rate=16000):
    signal, fs = torchaudio.load(wav_file)
    if fs != sample_rate:
        transform = torchaudio.transforms.Resample(fs, sample_rate)
        signal = transform(signal)
    # assert fs == 16000, fs
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        embeddings = F.normalize(embeddings, dim=2)
        embeddings = embeddings.squeeze().cpu().numpy()
    assert embeddings.shape[0] == size_embed, embeddings.shape[0]
    return embeddings

def verify_ref_path(reference_path, keywords, num_wavs):
    if os.path.isdir(reference_path):
        reference_wavpaths = sorted(glob.glob(os.path.join(reference_path, '*.wav')))
        reference_wavpaths = filter_path(reference_wavpaths, keywords)
        num_ref_wavs = len(reference_wavpaths)
        assert num_ref_wavs == num_wavs, \
            '# of input files ({}) and # of reference files ({}) mis-match!'.format(num_wavs, num_ref_wavs)
    elif os.path.isfile(reference_path):
        reference_wavpaths = [reference_path] * num_wavs
    else:
        raise Exception('reference path {} should be either a dir or a file'.format(reference_path))
    return reference_wavpaths    

def parse_args():
    usage = 'convert audio segments from source voice to target voice'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--input-path', type=str, help='input wav path')
    parser.add_argument('--keywords', type=str, help='filtered keywords separated by comma')
    parser.add_argument('--reference-path', type=str, help='reference wav path, either a dir or a file')
    parser.add_argument('--output-path', type=str, help='output wav path')
    parser.add_argument('--appendix', type=str, help='appendix attached to the output filename')
    parser.add_argument('--speaker-embed', type=str, required=True,
        choices=["speechbrain/spkrec-xvect-voxceleb", "speechbrain/spkrec-ecapa-voxceleb"],
        help="Pretrained model for extracting speaker emebdding.")
    parser.add_argument('--num-lim', type=int, help='# of max segments processed')    

    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # recording_id = 'MARCHE_AssessmentTacticalEnvironment'

    # # set appendix to attach to output file name indicating version
    # args.appendix = 'v6'

    # # set input path

    # # # v1, v2, v3 and v4
    # # voice = 'dmytro'
    # # stress = 'dictionary'
    # # wav_folder = '{}-{}'.format(voice, stress)
    # # args.input_path = os.path.join(work_path, 'outputs', 'sofw', 'espnet',
    # #     recording_id, wav_folder)

    # # v5 and v6
    # args.input_path = os.path.join(work_path, 'data', recording_id, 'segments')
    
    # # set reference path

    # # args.reference_path = os.path.join(work_path, 'data', recording_id, 'segments') # v1
    # # args.reference_path = os.path.join(work_path, 'data', recording_id, 'segments_all.wav') # v2

    # ref_voice = 'lada'
    # ref_stress = 'dictionary'
    # ref_wav_folder = '{}-{}'.format(ref_voice, ref_stress)
    # # args.reference_path = os.path.join(work_path, 'outputs', 'sofw', 'espnet',
    # #     recording_id, ref_wav_folder) # v3 and v5
    # args.reference_path = os.path.join(work_path, 'outputs', 'sofw', 'espnet',
    #      recording_id, '{}_all.wav'.format(ref_wav_folder)) # v4 and v6

    # # set other arguments
    # args.keywords = '.16000,_converted,_ukr,_resampled,_paired,_unpaired'
    # args.output_path = args.input_path
    # args.speaker_embed = "speechbrain/spkrec-xvect-voxceleb"
    # args.num_lim = 5

    # localize input arguments
    input_path = args.input_path
    keywords = args.keywords.split(',')
    reference_path = args.reference_path
    output_path = args.output_path
    appendix = args.appendix
    speaker_embed = args.speaker_embed
    num_lim = args.num_lim

    # sanity check
    assert os.path.isdir(input_path), \
        'input wav path: {} does not exist!'.format(iput_path)
    assert os.path.isdir(reference_path) or os.path.isfile(reference_path), \
        'reference path: {} does not exist!'.format(reference_path)  

    # generate output path (if needed)
    if os.path.isdir(output_path):
        print('using existing output dir: {}'.format(output_path))
    else:
        os.makedirs(output_path)
        print('created output dir: {}'.format(output_path))

    # print input arguments
    print('input dir: {}'.format(input_path))
    print('filtered keywords: {}'.format(', '.join(keywords)))
    print('reference dir/file: {}'.format(reference_path))
    print('output dir: {}'.format(output_path))
    print('appendix: {}'.format(appendix))
    print('speaker embedding model: {}'.format(speaker_embed))
    print('num of max segments processed: {}'.format(num_lim))

    # load voice conversion models
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
    model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # load speaker embedding (xvector) model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source=speaker_embed, run_opts={"device": device}, savedir=os.path.join('/tmp', args.speaker_embed))
    size_embed = spk_model[speaker_embed]
    print('speaker embedding dimension for {}: {}'.format(speaker_embed, size_embed))

    # get the input wav files
    input_wavpaths = sorted(glob.glob(os.path.join(input_path, '*.wav')))
    input_wavpaths = filter_path(input_wavpaths, keywords)
    num_wavs = len(input_wavpaths)
    print('# of wav files: {}'.format(num_wavs))

    # get the reference wav files
    reference_wavpaths = verify_ref_path(reference_path, keywords, num_wavs)

    num_processed = min(num_wavs, num_lim)

    for i in range(num_processed):

        input_wavpath = input_wavpaths[i]
        ref_wavpath = reference_wavpaths[i]

        # load the input wav file (resample if needed)
        y1, sr1 = librosa.load(input_wavpath)
        if sr1 != sample_rate:
            y1 = librosa.resample(y1, orig_sr=sr1, target_sr=sample_rate)
        inputs = processor(audio=y1, sampling_rate=sample_rate, return_tensors="pt")

        embeddings = f2embed(ref_wavpath, classifier, size_embed, sample_rate=sample_rate)
        embeddings = torch.from_numpy(embeddings).reshape(1, size_embed)

        try:
            # print('try here')
            length_input = len(y1)
            y2 = model.generate_speech(inputs["input_values"], embeddings, vocoder=vocoder)
        except RuntimeError:
            y1 = y1[1:-1] # slightly modify y1 to avoid error
            length_input_new = len(y1)
            print('{}/{} (i={}): encounting RuntimeError, slightly mofify the input length ({} -> {}), and retry ...'.format(
                i+1, num_wavs, i, length_input, length_input_new))
            inputs = processor(audio=y1, sampling_rate=sample_rate, return_tensors="pt")
            y2 = model.generate_speech(inputs["input_values"], embeddings, vocoder=vocoder)
        # else:
        #     y2 = model.generate_speech(inputs["input_values"], embeddings, vocoder=vocoder)
        y2 = y2.numpy()

        wavname = os.path.basename(input_wavpath)
        output_wavpath = os.path.join(output_path, wavname.replace('.wav', '_converted-{}.wav'.format(appendix)))
        sf.write(output_wavpath, y2, samplerate=sample_rate)
        print('wrote the converted wav file: {}'.format(output_wavpath))
