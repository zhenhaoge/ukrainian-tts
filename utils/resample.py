# resample wav files
#
# Zhenhao Ge, 2024-06-05

import os
from pathlib import Path
import glob
import librosa
import soundfile as sf

target_sr = 16000

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

recording_id = 'MARCHE_AssessmentTacticalEnvironment'
voice = 'oleksa' # ['tetiana', 'mykyta', 'lada', 'dmytro', 'oleksa']
stress = 'dictionary'
data_dir = os.path.join(work_path, 'outputs', 'sofw', 'espnet', recording_id, '{}-{}'.format(voice, stress))
assert os.path.isdir(data_dir), 'data dir: {} does not exist!'.format(data_dir)

input_wavfiles = sorted(glob.glob(os.path.join(data_dir, '*.wav')))
input_wavfiles = [f for f in input_wavfiles if '_new' not in f and '_resampled' not in f]
num_wavfiles = len(input_wavfiles)

for i in range(num_wavfiles):

    f1 = input_wavfiles[i]

    y, sr = librosa.load(f1)
    y2 = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    f2 = f1.replace('.wav', '.{}.wav'.format(target_sr))
    sf.write(f2, y2, target_sr, "PCM_16", format="wav")
    print('wrote {}'.format(f2))

