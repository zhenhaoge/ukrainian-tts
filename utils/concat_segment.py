# Concatenate the extracted audio segments to a single wav file
# Optionally, this script also save the speaker embedding of the concatenated segment, and
# compute the similarity to another version of the averaged speaker embedding obtained from
# espnet/egs2/TEMPLATE/tts1/zge/extract_spk_embed.py, and prove they are the same
# 
# Zhenhao Ge, 20244-06-10 

import os
from pathlib import Path
import argparse
import glob
import librosa
import numpy as np
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier
import torch.nn.functional as F
import torch
import torchaudio
from kaldiio import load_ark

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

def f2embed(wav_file, classifier, size_embed):
    signal, fs = torchaudio.load(wav_file)
    assert fs == 16000, fs
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        embeddings = F.normalize(embeddings, dim=2)
        embeddings = embeddings.squeeze().cpu().numpy()
    assert embeddings.shape[0] == size_embed, embeddings.shape[0]
    return embeddings    

def parse_args():
    usage = "usage: concatenate audio segments"
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--wav-path', type=str, \
        help='input wav path contains multiple wav files')
    parser.add_argument('--keywords', type=str, \
        help="list of keywords seperated by comma to filter out, e.g., '16000', '_new'")
    parser.add_argument('--concat-wavpath', type=str,
        help='concatenated wav file path')
    parser.add_argument('--speaker-embed', type=str, \
        default="speechbrain/spkrec-xvect-voxceleb", help='speaker embedding model')
    return parser.parse_args()

if __name__ == '__main__': 

    # runtime mode
    args = parse_args()

    # # interacitve mode
    # args = argparse.ArgumentParser()

    # # # example 1
    # # recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    # # args.wav_path = os.path.join(work_path, 'data', recording_id, 'segments') 
    # # args.concat_wavpath = os.path.join(work_path, 'data', recording_id, 'segments_all.wav')

    # # example 2
    # recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    # voice = 'oleksa' # dmytro, lada, mykyta, oleksa, tetiana
    # stress = 'dictionary'
    # spk_folder = '{}-{}'.format(voice, stress)
    # args.wav_path = os.path.join(work_path, 'outputs', 'sofw', 'espnet', recording_id, spk_folder)
    # args.concat_wavpath = os.path.join(work_path, 'outputs', 'sofw', 'espnet', recording_id, '{}_all.wav'.format(spk_folder))

    # args.keywords = '.16000,_new,_converted,_paired,_unpaired'
    # args.speaker_embed = "speechbrain/spkrec-xvect-voxceleb"

    # localize input arguments
    wav_path = args.wav_path
    keywords = args.keywords.split(',')
    concat_wavpath = args.concat_wavpath
    speaker_embed = args.speaker_embed

    # sanity check
    assert os.path.isdir(wav_path), 'wav dir: {} does not exist!'.format(wav_path)

    # print the input arguments
    print('wav path: {}'.format(wav_path))
    print('filtered keywords: {}'.format(', '.join(keywords)))
    print('output concatnated segment: {}'.format(concat_wavpath))
    print('speaker embedding model: {}'.format(speaker_embed))

    # get wav filepaths
    wav_filepaths = sorted(glob.glob(os.path.join(wav_path, '*.wav')))
    wav_filepaths = filter_path(wav_filepaths, keywords)
    num_wavs = len(wav_filepaths)
    print('# of wav files: {}'.format(num_wavs))

    # concatenate audio segments
    y = np.array([])
    for i, wav_file in enumerate(wav_filepaths):
        y0, sr = librosa.load(wav_file)
        y = np.append(y, y0)

    # write concatenated segment
    sf.write(concat_wavpath, y, sr)
    print('wrote the concatenated segment: {}'.format(concat_wavpath))

    #%% additional tasks

    # optional task 1: generate average speaker embedding from the concatenated segment

    # load speaker embedding (xvector) model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source=speaker_embed, run_opts={"device": device}, savedir=os.path.join('/tmp', args.speaker_embed))
    size_embed = spk_model[speaker_embed]

    # resample the original concatenated segment to 16000 (if needed)
    if sr != sample_rate:
        concat_wavpath2 = concat_wavpath.replace('.wav', '.{}.wav'.format(sample_rate))
        y2 = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
        sf.write(concat_wavpath2, y2, sample_rate)
        print('wrote resampled wav file: {}'.format(concat_wavpath2))
    else:
        concat_wavpath2 = concat_wavpath
        y2 = y1

    # get speaker embedding of the concatenated segment    
    embeddings = f2embed(concat_wavpath2, classifier, size_embed)

    # save to .npy file
    concat_npypath = concat_wavpath2.replace('.wav', '.xvector-{}.npy'.format(size_embed))
    np.save(concat_npypath, embeddings)
    print('saved embedding of the concatenated segment: {}'.format(concat_npypath))

    # # optional task 2: compare with another version of averaged speaker embedding

    # # get another version of the overall averaged speaker embedding of the English speaker in the sample recording
    # concat_arkpath = os.path.abspath(os.path.join(args.concat_wavpath, os.path.pardir,
    #     'xvectors_512', 'spk_xvector.ark'))
    # assert os.path.isfile(concat_arkpath), \
    #     'concatenated ark path: {} does not exist!'.format(concat_arkpath)  
    # xvectors = {k: v for k, v in load_ark(concat_arkpath)}
    # embeddings2 = xvectors['0']

    # # compute the cosine similarity score
    # embeddings_tensor = torch.from_numpy(embeddings)
    # embeddings2_tensor = torch.from_numpy(embeddings2)
    # score = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(embeddings_tensor, embeddings2_tensor)
    # print('cosine similarity score compared to {}: {:.4f}'.format(concat_arkpath, score[0].item()))
