# use spkrec-ecapa-voxceleb xvector model to generate tts voice of the target speaker
#
# method:
#   - use the eng segment as the reference audio, and generate the ukr syntheized output audio
# 
# conclusion:
#   - current framework does not generalize well to the unseen speaker, even though using the same
#     xvector model
#   - The system has already done its best to pick up the most close voice, but it definitely sounds
#     like one of the 5 in-domain seen speaker's voice
#   - the avg. speaker similarity score is only 0.11 (>0.7 can be considered to be the same speaker)
# 
#  Zhenhao Ge, 2024-06-05

import os
from pathlib import Path
import glob
import argparse
import torchaudio
import torch
import numpy as np
import soundfile as sf
from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

from ukrainian_tts.tts import TTS, Voices, Stress
from ukrainian_tts.formatter import preprocess_text
from ukrainian_tts.stress import sentence_to_stress, stress_dict, stress_with_model
from sofw.utils import read_trans, parse_fid

def filter_path(paths, keywords):
    for kw in keywords:
        paths = [f for f in paths if kw not in f]
    return paths    

def synthesize(text, spembs, output_file):

    output = tts.synthesizer(text, spembs=spembs)
    # print('output keys: {}'.format(list(output.keys())))

    wav = output['wav']
    wav = wav.view(-1).cpu().numpy()
    sf.write(output_file, wav, tts.synthesizer.fs, subtype="PCM_16", format="wav")
    print('output wav: {}'.format(output_file))

    return wav

cache_folder = os.path.join(work_path, 'model', 'espnet')
assert os.path.isdir(cache_folder), 'dir: {} does not exist!'.format(cache_folder)
print('model folder: {}'.format(cache_folder))

# set device
# if device=cpu, export OMP_NUM_THREADS=1 if use single CPU
device = 'cuda:1' # options: cpu, cuda, mps, etc.

# load TTS model
tts = TTS(device=device, cache_folder=cache_folder)

# show stress options
stresses = [stress for stress in Stress]
print('{} stresses available:'.format(len(stresses)))
stresses

# select stress option
stress_type = 'dictionary'
stress = stresses[0]
print('selected stress type: {}'.format(stress.value))

# determine if stress is on (true if stress option is model)
if stress.value == Stress.Model.value:
    stress = True
else:
    stress = False
print('stress is on? {}'.format(stress))

# show voice options

recording_id = 'MARCHE_AssessmentTacticalEnvironment'
speaker_path = os.path.join(work_path, 'data', recording_id, 'segments')
assert os.path.isdir(speaker_path), 'speaker dir: {} does not exist!'.format(speaker_path)

# get the list of sentences (id, text) from the transcription file
sentences = read_trans(trans_file)
num_sents = len(sentences)
print('there are {} sentences in transcription file {}'.format(num_sents, trans_file))

speaker_filepaths = sorted(glob.glob(os.path.join(speaker_path, '*.wav')))
keywords = ['_ukr', '_new', '_resampled', '16000']
speaker_filepaths = filter_path(speaker_filepaths, keywords=keywords)
num_spkr_files = len(speaker_filepaths)
print('# of speaker files: {}'.format(num_spkr_files))

trans_path = '/home/splola/kathol/SOFW/StaticVideos/scripts'
trans_file = os.path.join(trans_path, '{}.ukr.cor.sentids'.format(recording_id))
assert os.path.isfile(trans_file), 'transcription file: {} does not exist!'.format(trans_file)

assert num_spkr_files == num_sents, '#ref speaker files and #sentences mis-match!'

source_path = 'speechbrain/spkrec-ecapa-voxceleb'
model_saved_path = os.path.join(work_path, 'model', 'speechbrain', 'spkrec-ecapa-voxceleb')
classifier = EncoderClassifier.from_hparams(source=source_path, savedir=model_saved_path)
verification = SpeakerRecognition.from_hparams(source=source_path, savedir=model_saved_path)

# output_path = os.path.join(work_path, 'outputs', 'exp0', voice)
output_path = speaker_path
if os.path.isdir(output_path):
    print('use existing output dir: {}'.format(output_path))
else:
    os.makedirs(output_path)
    print('created new output dir: {}'.format(output_path))

target_sr = 16000
num_lim = 10
num_processed = min(num_sents, num_lim)
scores = [0 for _ in range(num_processed)]
predictions = [False for _ in range(num_processed)]

for i in range(num_processed):

    print('processing sentence {}/{} ...'.format(i+1, num_processed))

    # get fid and text from sentence tuple
    fid, text = sentences[i]

    # text preprocessing and stress decoration
    text = preprocess_text(text)
    text = sentence_to_stress(text, stress_with_model if stress else stress_dict)

    speaker_filepath = speaker_filepaths[i]
    assert os.path.isfile(speaker_filepath), 'speaker path: {} does not exist!'.format(speaker_filepath)
    y, sr = torchaudio.load(speaker_filepath)
    if sr != target_sr:
        y2 = librosa.resample(y.numpy(), orig_sr=sr, target_sr=target_sr)
        y2 = torch.from_numpy(y2)
    else:
        y2 = y

    spemb = classifier.encode_batch(y2, normalize=False)
    spemb_reshaped = spemb.squeeze() # shape: torch.Size([192])
    spemb_np = spemb_reshaped.numpy()

    # if voice in voice_list:
    #     spemb0_np = tts.xvectors[voice][0]
    #     spemb0 = torch.from_numpy(spemb0_np) # shape: torch.size([192])
    #     score0 = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(spemb_reshaped, spemb0)
    #     print('score between enrolled xvector and file-based xvector (score0): {:.2f}'.format(score0))

    output_file = os.path.join(output_path, '{}_ukr.wav'.format(fid))
    wav = synthesize(text, spemb_np, output_file)

    # score, prediction = verification.verify_files(speaker_filepath, output_file)
    waveform_x = verification.load_audio(speaker_filepath) # waveform_x.shape: [60930]
    waveform_y = verification.load_audio(output_file) # waveform_y.shape: [60930]
    print('waveform_x.shape: {}, waveform_y.shape: {}'.format(waveform_x.shape, waveform_y.shape))
    compare_ref_out = torch.equal(waveform_x, waveform_y)
    if compare_ref_out == True:
        print('{}/{}: output is the same as reference! referece: {} = output: {}'.format(
            i, num_processed, speaker_file, output_file))
    
    # wavform_x, sr_x = torchaudio.load(speaker_filepath) # wavform_x.shape: [83968], sr_x: 22050 
    # wavform_y, sr_y = torchaudio.load(output_file) # wavform_y.shape: [112128], sr_y: 22050
    # print('wavform_x.shape: {}, wavform_y.shape: {}'.format(wavform_x.shape, wavform_y.shape))

    # test_file1 = os.path.join(output_path, 'test_waveform_x.wav')
    # sf.write(test_file1, waveform_x, 16000)
    # print(test_file1)

    # test_file2 = os.path.join(output_path, 'test_waveform_y.wav')
    # sf.write(test_file2, waveform_x, 16000)
    # print(test_file2)

    # output_file2 = output_file.replace('.wav', '_resampled.wav')
    # wav2 = verification.load_audio(output_file)
    # sf.write(output_file2, wav2, 16000)
    # print(output_file2)

    scores[i], predictions[i] = verification.verify_files(speaker_filepath, output_file)

    # signal2, fs = torchaudio.load(output_file2)
    # spemb2 = classifier.encode_batch(signal2, normalize=False)
    # score = verification.similarity(spemb, spemb2)

# get avg. score (tetiana: 0.94, mykyta: 0.93, lada: 0.95, dmytro: 0.92, oleksa: 0.94)
score_list = [float(s[0].numpy()) for s in scores[:i]]
score_mean = np.mean(score_list)
print('mean similarity score for speaker in recording {}: {:.2f}'.format(recording_id, score_mean))

# get prediction accuracy (tetiana: 100%, mykyta: 100%, lada: 100%, dmytro: 100%, oleksa: 100%)
prediction_list = [int(p) for p in predictions[:i]]
prediction_mean = np.mean(prediction_list)
print('prediction accuracy for speaker in recording {}: {:.2f}%'.format(recording_id, prediction_mean*100))
