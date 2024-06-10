# Demo for Facebook's MMS-TTS-UKR (huggingface version)
#
# reference: https://huggingface.co/facebook/mms-tts-ukr
# maybe releated: Optimizing Inference Performance of Transformers on CPUs (https://arxiv.org/pdf/2102.06621)
#
# Zhenhao Ge, 2024-05-17

import os
from pathlib import Path
import torch
from transformers import VitsModel, AutoTokenizer
import time
# import scipy
import soundfile as sf
import numpy as np
import subprocess

# to enforce single-CPU run, the following two commands don't help
# os.environ["OMP_NUM_THREADS"] = '1'
# os.environ["MKL_NUM_THREADS"] = '1'
# you need to do the following export in the command line before running this script:
#   export OMP_NUM_THREADS=1
# once finish, then unset OMP_NUM_THREADS

def get_hostname():
    hostname = subprocess.check_output('hostname').decode('ascii').rstrip()
    return hostname

hostname = get_hostname()
print('hostname: {}'.format(hostname))

# set paths
home_path = str(Path.home())
work_path = os.path.join(home_path, 'code', 'repo', 'ukr-tts')
if os.getcwd() != work_path:
    os.chdir(work_path)
print('current path: {}'.format(os.getcwd()))

output_path = os.path.join(work_path, 'outputs')
print('output path: {}'.format(output_path))

def synthesize(text, device='cpu'):
    inputs = tokenizer(text, return_tensors="pt")

    # move inputs from cpu to cuda if needed
    if inputs['input_ids'].device.type == 'cpu' and 'cuda' in device:
        inputs['input_ids'].to(device)
    if inputs['attention_mask'].device.type == 'cpu' and 'cuda' in device:
        inputs['attention_mask'].to(device)    

    with torch.no_grad():
        output = model(**inputs).waveform
    wav = np.ravel(np.array(output)) 

    return wav

device = 'cpu'
# device = 'cuda:1'

# load model and tokenizer (vocab-size: 39)
# (they are loaded to cpu by default)
model = VitsModel.from_pretrained("facebook/mms-tts-ukr")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ukr")
print('model device: {}'.format(model.device))

# # this setting will lead to a different model
# # it is on cpu, but it is not the same as the default model above (will cause nan output)
# model = VitsModel.from_pretrained("facebook/mms-tts-ukr", device_map=device)
# tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ukr", device_map=device)

#%%  inference - single file 
text = "Привіт, як у тебе справи?"
start = time.time()
wav = synthesize(text, device)
end = time.time()
dur_proc = end - start
dur_out = len(wav) / model.config.sampling_rate
rtf = dur_proc / dur_out
print('RTF: {:.3f} ({:.3f} / {:.3f})'.format(rtf, dur_proc, dur_out))

output_file = os.path.join(output_path, "short.wav")
# scipy.io.wavfile.write(output_file, rate=model.config.sampling_rate, data=output)
sf.write(output_file, wav, model.config.sampling_rate)
print('output wav: {}'.format(output_file))

#%% inference - multiple files

text = "К+ам'ян+ець-Под+ільський - м+істо в Хмельн+ицькій +області Укра+їни, ц+ентр Кам'ян+ець-Под+ільської міськ+ої об'+єднаної територі+альної гром+ади +і Кам'ян+ець-Под+ільського рай+ону."
num_reps = 10
rtfs = [0 for _ in range(num_reps)]
for i in range(num_reps):

    start = time.time()
    wav = synthesize(text, device)
    end = time.time()
    dur_proc = end - start
    dur_out = len(wav) / model.config.sampling_rate
    rtf = dur_proc / dur_out
    print('RTF: {:.3f} ({:.3f} / {:.3f})'.format(rtf, dur_proc, dur_out))

    output_file = os.path.join(output_path, "long_{:02d}.wav".format(i))
    sf.write(output_file, wav, model.config.sampling_rate)
    print('output wav: {}'.format(output_file))

    rtfs[i] = rtf

print(f"avg. RTF = {np.mean(rtfs):5f}") 

