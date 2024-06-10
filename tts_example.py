import os
from pathlib import Path
import numpy as np

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

output_path = os.path.join(work_path, 'outputs')
print('output path: {}'.format(output_path))

from ukrainian_tts.tts import TTS, Voices, Stress

cache_folder = os.path.join(work_path, 'model', 'espnet')
assert os.path.isdir(cache_folder), 'dir: {} does not exist!'.format(cache_folder)
print('model folder: {}'.format(cache_folder))

# set device
# if device=cpu, export OMP_NUM_THREADS=1 if use single CPU
device = 'cuda:0' # options: cpu, cuda, cuda:x, mps, etc.

# load TTS model
tts = TTS(device=device, cache_folder=cache_folder) 
# tts = TTS(device=device)

# show voice options
voices = [voice for voice in Voices]
print('{} voices available:'.format(len(voices)))
voices

# show stress options
stresses = [stress for stress in Stress]
print('{} stresses available:'.format(len(stresses)))
stresses

#%% inference - single file

# set voice
# voice = Voices.Dmytro.value
voice = Voices.Tetiana.value

# set stress
stress = Stress.Dictionary.value

# synthesize speech
text = "Привіт, як у тебе справи?"
output_file = os.path.join(output_path, 'short_{}_{}.wav'.format(voice, stress))
with open(output_file, mode="wb") as file:
    output_fp, output_text, rtf = tts.tts(text, voice, stress, file)
print("Accented text:", output_text)
print(f"RTF = {rtf:5f}")

#%% inference - multiple files

text = "К+ам'ян+ець-Под+ільський - м+істо в Хмельн+ицькій +області Укра+їни, ц+ентр Кам'ян+ець-Под+ільської міськ+ої об'+єднаної територі+альної гром+ади +і Кам'ян+ець-Под+ільського рай+ону."
rtfs = []
for i, voice in enumerate(voices):
    for j, stress in enumerate(stresses):
        print('voice {}: {}, stress {}: {}'.format(
            i, voice.value, j, stress.value))
        output_file = os.path.join(output_path, 'long_{}_{}.wav'.format(voice.value, stress.value))
        with open(output_file, mode="wb") as file:
            output_fp, output_text, rtf = tts.tts(text, voice.value, stress.value, file)
        rtfs.append(rtf)    
        print("Accented text:", output_text)
        print(f"RTF = {rtf:5f}")

print(f"avg. RTF = {np.mean(rtfs):5f}")

