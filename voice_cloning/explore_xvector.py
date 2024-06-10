# voice cloning example
#
# reference:
#  - use speechbrain for xvector: https://huggingface.co/speechbrain/spkrec-xvect-voxceleb
#  - use speechbrain for ecapa xvector: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
#
# Zhenhao Ge, 2024-06-04

import os
from pathlib import Path
import argparse
from kaldiio import load_ark

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

model_path = os.path.join(work_path, 'model', 'espnet')
speakers_filepath = os.path.join(model_path, "spk_xvector.ark")
assert os.path.isfile(speakers_filepath), \
    'speaker file {} does not exist!'.format(speakers_filepath)

xvectors = {k: v for k, v in load_ark(speakers_filepath)}
speakers = sorted(xvectors.keys())
print('speakers: {}'.format(', '.join(speakers)))

dim_xvector = xvectors[speakers[0]].shape
print('xvector dim: {}'.format(dim_xvector)) # 1 X 192

# option 1: using espnet/utils/syn_wav.sh (dim 512)
espnet_dir = '/home/users/zge/code/repo/espnet'
ref_speaker_filepath = os.path.join(espnet_dir, 'egs/libritts/tts1/decode/example/xvectors', 'spk_xvector.ark')
assert os.path.isfile(ref_speaker_filepath), \
    'ref speaker file: {} does not exist!'.format(ref_speaker_filepath)
tmp = load_ark(ref_speaker_filepath)
ref_xvectors = {k: v for k, v in load_ark(ref_speaker_filepath)}

# option 2: using espnet/egs2/TEMPLATE/tts1/zge/extract_spk_embed.py with spkrec-xvector-voxceleb (dim 512)
recording_id = 'MARCHE_AssessmentTacticalEnvironment'
ref_speaker_filepath = os.path.join(work_path, 'data', recording_id, 'xvectors_512', 'xvector.ark')
for k,v in load_ark(ref_speaker_filepath):
    break

# option 3: using epsnet/egs2/TEMPLATE/tts1/zge/extract_spk_embed.py with spkrec-ecapa-voxceleb (dim 192)
recording_id = 'MARCHE_AssessmentTacticalEnvironment'
ref_speaker_filepath = os.path.join(work_path, 'data', recording_id, 'xvectors', 'xvector.ark')
for k,v in load_ark(ref_speaker_filepath):
    break

