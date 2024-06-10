# demo for using speechbrain ecapa xvector model trained from voxceleb
#
# references:
#  - speechbrain 
#  - speechbrain inference docs: https://speechbrain.readthedocs.io/en/0.5.7/API/speechbrain.pretrained.interfaces.html
# Zhenhao Ge, 2024-06-05

import os
from pathlib import Path

import torch
import torchaudio
# from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

source_path = 'speechbrain/spkrec-ecapa-voxceleb'
model_saved_path = os.path.join(work_path, 'model', 'speechbrain', 'spkrec-ecapa-voxceleb')
classifier = EncoderClassifier.from_hparams(source=source_path, savedir=model_saved_path)
verification = SpeakerRecognition.from_hparams(source=source_path, savedir=model_saved_path)

recording_id = 'MARCHE_AssessmentTacticalEnvironment'

# get one speaker embedding
wavfile1 = os.path.join(work_path, 'data', recording_id, 'segments', '0001_0.0_2.3.wav')
assert os.path.isfile(wavfile1), 'wav file: {} does not exist!'.format(wavfile1)
signal1, fs = torchaudio.load(wavfile1)
spemb1 = classifier.encode_batch(signal1, normalize=True)
spemb1_np = spemb1.squeeze().numpy()

# get another speaker embedding
wavfile2 = os.path.join(work_path, 'data', recording_id, 'segments', '0002_2.16_5.26.wav')
assert os.path.isfile(wavfile2), 'wav file: {} does not exist!'.format(wavfile2)
signal2, fs = torchaudio.load(wavfile2)
spemb2 = classifier.encode_batch(signal2, normalize=True)
spemb2_np = spemb2.squeeze().numpy()

#%% multiple methods to computer the similarity score

# compute the speaker cosine similarity score using wav files
score, prediction = verification.verify_files(wavfile1, wavfile2)

# compute the speaker cosine similarity score using embeddings (from verification module)
score2 = verification.similarity(spemb1, spemb2)

# computer the speaker cosine similarity score using embeddings (from torch)
score3 = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(spemb1, spemb2)
