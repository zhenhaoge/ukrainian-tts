import os
import subprocess
from glob import glob

FFMPEG_BIN = '/usr/bin/ffmpeg'

SAMPLE_RATE = 22050
FILE_FORMAT = 'wav'

DATA_ROOT = '/home/users/zge/data1/datasets/ukrainian-tts'
speaker = 'tetiana' # 'lada', 'tetiana', 'kateryna', 'mykyta', 'oleksa'
dataset = 'dataset_tetiana'
subtype = 'accept' # 'accept', 'reject', ''
FILES_MASK = os.path.join(DATA_ROOT, speaker, dataset, subtype, '*.ogg')
INPUT_DIR = os.path.dirname(FILES_MASK)
assert os.path.isdir(INPUT_DIR), 'input dir: {} does not exist!'.format(INPUT_DIR)
SAVE_TO = os.path.join(DATA_ROOT, speaker, '{}_22khz'.format(dataset), subtype)
if os.path.isdir(SAVE_TO):
    print('use existing output dir: {}'.format(SAVE_TO))
else:
    os.makedirs(SAVE_TO)
    print('created new output dir: {}'.format(SAVE_TO))

def resample(filename_full, verbose=False):
    filename = filename_full.split('/')[-1]
    out_filename = SAVE_TO + '/' + filename.replace('ogg', FILE_FORMAT)

    cmd = [FFMPEG_BIN, '-i', filename_full, '-ar', str(SAMPLE_RATE), out_filename]

    if verbose:
        print('Running {}'.format(' '.join(cmd)))

    if os.path.exists(out_filename):
        # print('Skipping')
        return
    else:
        result = subprocess.run(cmd)
        print(result)

def run():
    input_files = glob(FILES_MASK)
    nfiles_input = len(input_files)

    print('Found {} files'.format(nfiles_input))

    for i, f in enumerate(input_files):
        # resample(f, verbose=True)
        resample(f)

    # sanity check
    output_files = glob(os.path.join(SAVE_TO, '*.{}'.format(FILE_FORMAT)))
    nfiles_output = len(output_files)
    assert nfiles_input == nfiles_output, \
        '#files mismatch: input {} vs. output {}'.format(nfiles_input, nfiles_output)

if __name__ == '__main__':
    run()
