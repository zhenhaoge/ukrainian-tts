# Demo of SpeechT5 Voice Conversion
#
# References:
#  - speecht5_vc on huggingface: https://huggingface.co/microsoft/speecht5_vc
#  - get xvector_speaker_embedding: https://huggingface.co/microsoft/speecht5_vc/discussions/5
#
# Zhenhao Ge, 2024-06-10

import os
from pathlib import Path
import glob
import shutil

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
nsamples_input = len(dataset)
print('# of input audio files to select from: {}'.format(nsamples_input))

# get the input speech
id_input = 0
example_speech = dataset[id_input]["audio"]["array"]
fid_input = dataset[id_input]['id']

# write the input text file
text_filepath = os.path.join(work_path, 'voice_conversion', '01_input_{}.txt'.format(fid_input))
with open(text_filepath, 'w') as f:
    f.writelines('{}\n'.format(dataset[id_input]['text']))
print('wrote input txt file: {}'.format(text_filepath))

# get input audio sample rate
sampling_rate = dataset.features["audio"].sampling_rate

# write the example speech
input_filename = "01_input_{}.wav".format(fid_input)
input_filepath = os.path.join(work_path, 'voice_conversion', input_filename)
sf.write(input_filepath, example_speech, samplerate=sampling_rate)
print('wrote input wav file: {}'.format(input_filepath))

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors="pt")

#%% load xvector

# load xvector containing speaker's voice characteristics from a file
import numpy as np
import torch

# # option 1: get speaker embedding from online dataset
# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0) # shape: 1 X 512

# option 2: get speaker embedding locally

# get the embedding path
embedding_path = os.path.join(home_path, 'code', 'repo', 'speecht5-vc', 'xvectors', 'cmu_arctic')
assert os.path.isdir(embedding_path), 'embedding dir: {} does not exist!'.format(embedding_path)
embedding_filepaths = glob.glob(os.path.join(embedding_path, '*.npy'))
nsamples_embedding = len(embedding_filepaths)
print('# of embedding files: {}'.format(nsamples_embedding))

# set the speaker and wav filename
spk = 'slt'
fid = 'a0508'
arctic_dataset = 'cmu_us_{}_arctic'.format(spk)
wavname = 'arctic_{}.wav'.format(fid)

# get the reference wav path
wav_path = os.path.join(home_path, 'data1', 'datasets', 'cmu_arctic', arctic_dataset, 'wav')
wav_filepaths = glob.glob(os.path.join(wav_path, '*.wav'))
nsamples_wav = len(wav_filepaths)
print('# of wav files: {}'.format(nsamples_wav))

# get the reference wav file
wav_filepath = os.path.join(wav_path, wavname)
assert os.path.isfile(wav_filepath), 'reference wav file: {} does not exist!'.format(wav_path)

# copy it to the output folder
fid_reference = '{}-{}'.format(spk, fid)
wav_filename = '02_reference_{}.wav'.format(fid_reference)
wav_filepath2 = os.path.join(work_path, 'voice_conversion', wav_filename)
shutil.copyfile(wav_filepath, wav_filepath2)
print('reference wav file: {}'.format(wav_filepath2))

# get the embedding file
embedding_filename = '-'.join([arctic_dataset, 'wav', wavname.replace('.wav', '.npy')])
embedding_file = os.path.join(embedding_path, embedding_filename)
assert os.path.isfile(embedding_file), 'embedding file: {} does not exist!'.format(embedding_file)
print('reference speaker embedding file: {}'.format(embedding_file))

# load in the speaker embedding
speaker_embeddings = np.load(embedding_file)
speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

#%% generate converted speech

# generate converted speech 
speech = model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=vocoder)

# write out the converted speech
fid_converted = '{}_{}'.format(fid_input, fid_reference)
outfile = os.path.join(work_path, 'voice_conversion', "03_converted_{}.wav".format(fid_converted))
sf.write(outfile, speech.numpy(), samplerate=sampling_rate)
print('output converted wav file: {}'.format(outfile))
