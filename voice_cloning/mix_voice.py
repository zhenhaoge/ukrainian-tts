# this script contains code to generate speech with a mixed voice
# however, the mixed voice sounds like one of the single-speaker's voice
# maybe, we can not do simple add and average, need to take a look how the
# xvector is generated
#
# Zhenhao Ge, 2024-05-20

import os
from pathlib import Path
import numpy as np
import soundfile as sf
from kaldiio import load_ark

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

output_path = os.path.join(work_path, 'outputs')
print('output path: {}'.format(output_path))

from ukrainian_tts.tts import TTS, Voices, Stress
from ukrainian_tts.formatter import preprocess_text
from ukrainian_tts.stress import sentence_to_stress, stress_dict, stress_with_model

def synthesize(text, spembs, output_file):

    output = tts.synthesizer(text, spembs=spembs)
    # print('output keys: {}'.format(list(output.keys())))

    wav = output['wav']
    wav = wav.view(-1).cpu().numpy()
    sf.write(output_file, wav, tts.synthesizer.fs, "PCM_16", format="wav")
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

#%% deal with text

# show stress options
stresses = [stress for stress in Stress]
print('{} stresses available:'.format(len(stresses)))
stresses

# select stress option
stress = stresses[0]

# determine if stress is on (true if stress option is model)
if stress.value == Stress.Model.value:
    stress = True
else:
    stress = False
print('stress is on? {}'.format(stress))

# set text
text = "Привіт, як у тебе справи?"

# preprocess text
text_ppd = preprocess_text(text)

# stress text based on stress
text_accented = sentence_to_stress(text_ppd, stress_with_model if stress else stress_dict)

# show the text processing
print('Original text: {}'.format(text))
print('Processed text: {}'.format(text_ppd))
print('Accented text: {}'.format(text_accented))

#%% mix existing voices

# show voice options
voices = [voice for voice in Voices]
print('{} voices available:'.format(len(voices)))
voices

# set the speaker indeces
# idx_sel = [2,4]
idx_sel = [0,1]

# select two voices based on indeces
voice_a  = voices[idx_sel[0]]
voice_b = voices[idx_sel[1]]

# speaker embedding percent
percent_a = 0.425
percent_b = 1 - percent_a

print('voice a: {} ({:.2f}%), voice b: {} ({:.2f}%)'.format(
    voice_a.value, percent_a*100, voice_b.value, percent_b*100))

# mix speaker embeddings
spembs_a = tts.xvectors[voice_a.value][0]
spembs_b = tts.xvectors[voice_b.value][0]
spembs_mixed = spembs_a * percent_a + spembs_b * percent_b

print('spembs_a[:10]: {}'.format(spembs_a[:10]))
print('spembs_b[:10]: {}'.format(spembs_b[:10]))
print('spembs_mixed[:10]: {}'.format(spembs_mixed[:10]))

# use the voice of the speaker a
output_file = os.path.join(output_path, 'spk-a_{}.wav'.format(voice_a.value))
wav_a = synthesize(text_accented, spembs_a, output_file)

# use the voice of the speaker b
output_file = os.path.join(output_path, 'spk-b_{}.wav'.format(voice_b.value))
wav_b = synthesize(text_accented, spembs_b, output_file)

# use the voice of the mixed speaker embedding
output_file = os.path.join(output_path, 'mix_{}-{}_{}-{}.wav'.format(
    voice_a.value, percent_a, voice_b.value, percent_b))
wav_mixed = synthesize(text_accented, spembs_mixed, output_file)

# observation: the mixed voice sounds like one of the single speaker voice, and it may switch
# from one to the other

#%% new voice (using the output generaeted by espnet/egs2/TEMPLATE/tts1/zge/extract_spk_embed.py)

recording_id = 'MARCHE_AssessmentTacticalEnvironment'
# speaker_path = os.path.join(work_path, 'data', recording_id, 'xvectors', 'spk_xvector.ark') # avg. embeddings
speaker_emb_path = os.path.join(work_path, 'data', recording_id, 'xvectors', 'xvector.ark') # all embeddings
assert os.path.isfile(speaker_emb_path), 'speaker embedding file {} does not exist!'.format(speaker_emb_path)
xvectors = {k:v for k, v in load_ark(speaker_emb_path)}
sids = sorted(xvectors.keys())
num_sids = len(sids)
print('# of sids: {}'.format(num_sids))
spembs_new = xvectors[sids[0]]
spembs_new_np = spembs_new.squeeze()

output_file = os.path.join(output_path, 'new.wav')
wav_new = synthesize(text_accented, spembs_new, output_file)

#%% new voice (directly using speechbrain pre-trained model)

import torchaudio
from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition

source_path = 'speechbrain/spkrec-ecapa-voxceleb'
model_saved_path = os.path.join(work_path, 'model', 'speechbrain', 'spkrec-ecapa-voxceleb')
classifier = EncoderClassifier.from_hparams(source=source_path, savedir=model_saved_path)
verification = SpeakerRecognition.from_hparams(source=source_path, savedir=model_saved_path)

recording_id = 'MARCHE_AssessmentTacticalEnvironment'
speaker_path = os.path.join(work_path, 'data', recording_id, 'segments', '0001_0.0_2.3.wav')
assert os.path.isfile(speaker_path), 'speaker file {} does not exist!'.format(speaker_path)
signal, fs = torchaudio.load(speaker_path)
spemb_new2 = classifier.encode_batch(signal, normalize=True)
spemb_new2_np = spemb_new2.squeeze().numpy()

# spemb_new2_norm = classifier.encode_batch(signal, normalize=True)
# spemb_new2_unnorm = classifier.encode_batch(signal, normalize=False)

# spemb_new2_norm_np = spemb_new2_norm.squeeze().numpy()
# spemb_new2_unnorm_np = spemb_new2_unnorm.squeeze().numpy()

# spemb_new2_norm_np[:10]
# spemb_new2_unnorm_np[:10]

output_file = os.path.join(output_path, 'new2.wav')
wav_new2 = synthesize(text_accented, spemb_new2_np, output_file)
