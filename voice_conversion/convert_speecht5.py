# Test the capability to convert the seen ukr-tts voice to an unseen voice using speecht5-vc
#
# Conclusions:
#   - voice is converted natually, but the prosody is changed due to the source and target language difference  
#
# Zhenhao Ge, 2024-06-10

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

def parse_args():
    usage = 'convert voice using speecht5-vc'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--input-wavpath', type=str, help='input wav filepath')
    parser.add_argument('--reference-filepath', type=str, help='reference filepath')
    parser.add_argument('--output-wavpath', type=str, help='output converted wav filepath')
    parser.add_argument('--speaker-embed', type=str, required=True,
        choices=["speechbrain/spkrec-xvect-voxceleb", "speechbrain/spkrec-ecapa-voxceleb"],
        help="Pretrained model for extracting speaker emebdding.")

    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # interactive mode
    args = argparse.ArgumentParser()

    voice = 'dmytro'
    stress = 'dictionary'
    recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    wav_folder = '{}-{}'.format(voice, stress)

    # get wav paths
    input_path = os.path.join(work_path, 'outputs', 'sofw', 'espnet',
        recording_id, wav_folder)
    import glob
    wavpaths = sorted(glob.glob(os.path.join(input_path, '*.wav')))
    keywords = ['_converted', '16000']
    wavpaths = filter_path(wavpaths, keywords=keywords)
    num_wavs = len(wavpaths)
    print('# of filtered wav files in {}: {}'.format(input_path, num_wavs))

    # set the input wav path by index
    idx = 1
    args.input_wavpath = wavpaths[idx]

    # set reference file (can either wav, or speaker embedding (npy, ark))

    # # option 1: wav reference file
    # wavname = os.path.basename(args.input_wavpath)
    # data_folder = os.path.join(work_path, 'data', recording_id, 'segments') 
    # args.reference_filepath = os.path.join(data_folder, wavname)

    # # option 2: ark reference file
    # data_folder = os.path.join(work_path, 'data', recording_id, 'xvectors_512')
    # args.reference_filepath = os.path.join(data_folder, 'spk_xvector.ark')

    # option 3: npy reference file
    data_folder = os.path.join(work_path, 'data', recording_id)
    args.reference_filepath = os.path.join(data_folder, 'segments_all.16000.xvector-512.npy')

    # set output wav path
    args.output_wavpath = args.input_wavpath.replace('.wav', '_converted.wav')

    # set speaker embedding model
    args.speaker_embed = "speechbrain/spkrec-xvect-voxceleb"

    # localize input arguments
    input_wavpath = args.input_wavpath
    reference_filepath = args.reference_filepath
    output_wavpath = args.output_wavpath
    speaker_embed = args.speaker_embed

    # check file existence
    assert os.path.isfile(input_wavpath), \
        'input wav file: {} does not exist!'.format(input_wavpath) 
    assert os.path.isfile(reference_filepath), \
        'reference file: {} does not exist!'.format(reference_filepath)

    # print input arguments
    print('input wav file: {}'.format(input_wavpath))
    print('reference file: {}'.format(reference_filepath))
    print('speaker embedding model: {}'.format(speaker_embed))

    # load voice conversion models
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
    model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # load speaker embedding (xvector) model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source=speaker_embed, run_opts={"device": device}, savedir=os.path.join('/tmp', args.speaker_embed))
    size_embed = spk_model[speaker_embed]
    print('speaker embedding dimension for {}: {}'.format(speaker_embed, size_embed))

    # load input audio
    y1, sr1 = librosa.load(input_wavpath)
    if sr1 != sample_rate:
        y1 = librosa.resample(y1, orig_sr=sr1, target_sr=sample_rate)
    inputs = processor(audio=y1, sampling_rate=sample_rate, return_tensors="pt")

    # extract the speaker embedding from the reference audio file
    ext = os.path.splitext(reference_filepath)[-1]
    if ext == '.wav':
        embeddings = f2embed(reference_filepath, classifier, size_embed)
    elif ext == '.npy':
        embeddings = np.load(reference_filepath)
    elif ext == '.ark':
        xvectors = {k: v for k, v in load_ark(reference_filepath)}
        embeddings = xvectors['0']
    spkr_emb_dim = embeddings.shape[-1]
    assert size_embed == spkr_emb_dim, 'speaker embedding dimension should be {}, but is {}'.format(
        size_embed, spkr_emb_dim)
    embeddings = torch.from_numpy(embeddings).reshape(1, size_embed)

    # generate the converted audio
    y2 = model.generate_speech(inputs["input_values"], embeddings, vocoder=vocoder)
    y2 = y2.numpy()

    sf.write(output_wavpath, y2, samplerate=sample_rate)
    print('wrote the converted wav file: {}'.format(output_wavpath))
