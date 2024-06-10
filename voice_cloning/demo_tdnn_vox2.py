import os
from pathlib import Path

import torch
import torchaudio
from speechbrain.pretrained.interfaces import Pretrained
from speechbrain.pretrained import EncoderClassifier

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

class Encoder(Pretrained):

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings,
                torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings


source_path = os.path.join(os.getcwd(), 'model', 'speechbrain', 'yangwang825-tdnn-vox2')
model_saved_path = os.path.join(os.getcwd(), 'model', 'speechbrain', 'yangwang825-tdnn-vox2')
# classifier = Encoder.from_hparams(source="yangwang825/tdnn-vox2", savedir=model_saved_path)
classifier = Encoder.from_hparams(source="yangwang825/tdnn-vox2", savedir=model_saved_path)

signal, fs = torchaudio.load('spk1_snt1.wav')
embeddings = classifier.encode_batch(signal)

