# verify if spkrec-ecapa-voxceleb is the model used to generate xvector in ukr-tts
#
# method:
#   - use the original (eng) or synthezied (ukr) audio segmented files as reference audio
#   - generate the parallel synthzied audio files and compare with the reference
#   - check the mean of speaker similarity cosine score across all pairs
#
# conclusion:
#   - the xvector model I found ('speechbrain/spkrec-ecapa-voxceleb') can generate xvectors, which
#     is similar to the xvector provided in the recipe, just need to downsample the reference speaker
#     audio to 16000
#   - the regenerated synthesized sample is in the same voice as the reference one (e.g., ***_new.wav is
#     in the same voice of ***.wav, which is used as the reference, for all 5 voices)
#   - check out the speaker similarity score and speaker recognition accuracy below at the bottom of this
#     script
#   - without using the downsamped reference audio, the output voice is not always the same, it can
#     switch from one to another
#
# references:
#   - speechbrain spkrec-ecapa-voxceleb model: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
#   - speechbrain spkrec-ecapa-voxceleb-mel-spec model: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb-mel-spec
# 
#  Zhenhao Ge, 2024-06-05

import os
from pathlib import Path
import glob
import librosa
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
from audio import wav_duration

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

def parse_args():
    usage = 'usage: verify if the xvector model can extract same or similar xvector, which ' + \
            'is comparable to the master xvector provided in the recipe'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--cache-folder', type=str, help='tts model path')
    parser.add_argument('--stress-type', type=str, help='stress type')
    parser.add_argument('--voice', type=str, help='voice: tetiana, mykyta, lada, dmytro, oleksa')
    parser.add_argument('--speaker-path', type=str, help='reference speaker path')
    parser.add_argument('--output-path', type=str, help='output syn file path')
    parser.add_argument('--trans-file', type=str, help='transcription file')
    parser.add_argument('--source-path', type=str, help='xvector model source path')
    parser.add_argument('--model-saved-path', type=str, help='xvector model saved path')
    parser.add_argument('--device', type=str, default='cpu', help='CPU/GPU decice')
    parser.add_argument('--appendix', type=str, default='new', help='appendix on the output file anme')
    return parser.parse_args()

if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # work_path = os.getcwd()
    # args.cache_folder = os.path.join(work_path, 'model', 'espnet')
    # args.stress_type = 'dictionary'
    # args.voice = 'tetiana' # ['tetiana', 'mykyta', 'lada', 'dmytro', 'oleksa']
    # recording_id = 'MARCHE_AssessmentTacticalEnvironment'
    # # args.speaker_path = os.path.join(work_path, 'outputs', 'sofw', 'espnet', recording_id,
    # #     '{}-{}'.format(args.voice, args.stress_type))
    # args.speaker_path = os.path.join(home_path, 'data1', 'datasets', 'ukrainian-tts',
    #     args.voice, 'dataset_{}_22khz'.format(args.voice))
    # args.output_path = os.path.join(work_path, 'outputs', 'sofw', 'espnet', recording_id,
    #     '{}-{}'.format(args.voice, args.stress_type))
    # trans_path = '/home/splola/kathol/SOFW/StaticVideos/scripts'
    # args.trans_file = os.path.join(trans_path, '{}.ukr.cor.sentids'.format(recording_id))
    # args.source_path = 'speechbrain/spkrec-ecapa-voxceleb'
    # args.model_saved_path = os.path.join(work_path, 'model', 'speechbrain', 'spkrec-ecapa-voxceleb')
    # # if device=cpu, export OMP_NUM_THREADS=1 if use single CPU
    # args.device = 'cuda:0' # options: cpu, cuda, mps, etc.
    # args.appendix = 'unpaired' # new, paired (same text), unpaired (different text)

    # show voice options
    voices = [voice for voice in Voices]
    voice_list = [voice.value for voice in voices]
    print('{} voices available: {}'.format(len(voices), ', '.join(voice_list)))

    # show stress options
    stresses = [stress for stress in Stress]
    print('{} stresses available:'.format(len(stresses)))
    stresses

    # localize input arguments
    cache_folder = args.cache_folder
    
    if args.stress_type == 'dictionary':
        stress = stresses[0]
    elif args.stress_type == 'model':
        stress = stresses[1]
    else:
        raise Exception('stress types can only be either dictionary or model')        
    voice = args.voice
    speaker_path = args.speaker_path
    output_path = args.output_path
    trans_file = args.trans_file
    source_path = args.source_path
    model_saved_path = args.model_saved_path
    device = args.device
    appendix = args.appendix

    # print input arguments
    print('selected stress type: {}'.format(stress.value))
    print('selected voice: {}'.format(voice))
    print('device: {}'.format(device))
    print('appendix: {}'.format(appendix))

    # check path existance
    assert os.path.isdir(cache_folder), 'dir: {} does not exist!'.format(cache_folder)
    print('model path: {}'.format(cache_folder))
    assert os.path.isdir(speaker_path), 'speaker dir: {} does not exist!'.format(speaker_path)
    print('speaker path: {}'.format(speaker_path))
    assert os.path.isdir(output_path), 'output dir: {} does not exist!'.format(output_path)
    print('output path: {}'.format(output_path))
    assert os.path.isfile(trans_file), 'transcription file: {} does not exist!'.format(trans_file)
    print('transcription file: {}'.format(trans_file))
    print('xvector model source path: {}'.format(source_path))
    print('xvector model saved path: {}'.format(model_saved_path))

    # load TTS model
    tts = TTS(device=device, cache_folder=cache_folder)

    # determine if stress is on (true if stress option is model)
    if stress.value == Stress.Model.value:
        stress = True
    else:
        stress = False
    print('stress is on? {}'.format(stress))

    # get the list of sentences (id, text) from the transcription file
    sentences = read_trans(trans_file)
    num_sents = len(sentences)
    print('there are {} sentences in transcription file {}'.format(num_sents, trans_file))

    # get reference speaker files
    speaker_filepaths = sorted(glob.glob(os.path.join(speaker_path, '**', '*.wav'), recursive=True))
    keywords = ['_new', '_paired', 'unpaired', '_resampled', '16000']
    speaker_filepaths = filter_path(speaker_filepaths, keywords=keywords)
    num_spkr_files = len(speaker_filepaths)
    print('# of the original speaker files: {}'.format(num_spkr_files))
    # assert num_spkr_files == num_sents, '#ref speaker files and #sentences mis-match!'

    if num_spkr_files < num_sents:
        raise Exception('# of reference speaker files should be greater or equal to the # of sentences')
    elif num_spkr_files == num_sents:
        print('# of the speaker files matches with # of the sentences')
        speaker_filepaths_sel = speaker_filepaths
    else:
        # get the audio files with longer durations as reference (more consistent speaker embedding) 
        print('getting durations of {} speaker files ...'.format(num_spkr_files))
        durs = [wav_duration(f) for f in speaker_filepaths]
        idx = np.argsort(durs)[::-1]
        dur_mean = np.mean([durs[i] for i in idx[:num_sents]])
        speaker_filepaths_sel = [f for i, f in enumerate(speaker_filepaths) if i in idx[:num_sents]]
        num_spkr_files = num_sents
        print('# of the selected speaker files: {}, with mean dur {:.2f} secs'.format(
            num_spkr_files, dur_mean))
    
    classifier = EncoderClassifier.from_hparams(source=source_path, savedir=model_saved_path)
    verification = SpeakerRecognition.from_hparams(source=source_path, savedir=model_saved_path)

    # output_path = os.path.join(work_path, 'outputs', 'exp0', voice)
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

        # load the reference audio (get or generate the downsampled version if needed)
        speaker_filepath = speaker_filepaths_sel[i]
        assert os.path.isfile(speaker_filepath), 'speaker path: {} does not exist!'.format(speaker_filepath)
        y, sr = torchaudio.load(speaker_filepath)
        if sr != target_sr:
            speaker_filepath2 = speaker_filepath.replace('.wav', '.{}.wav'.format(target_sr))
            if os.path.isfile(speaker_filepath2):
                # use the downsampled reference file (from 22050 to 16000)
                y2, sr2 = torchaudio.load(speaker_filepath2)
            else:    
                # generate the downsampled reference audio (from 22050 to 16000)
                y2 = librosa.resample(y.numpy(), orig_sr=sr, target_sr=target_sr)
                y2 = torch.from_numpy(y2)
            # y2 = librosa.resample(y.numpy(), orig_sr=sr, target_sr=target_sr)
            # y2 = torch.from_numpy(y2)
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

        output_file = os.path.join(output_path, '{}_{}.wav'.format(fid, appendix))
        wav = synthesize(text, spemb_np, output_file)

        # # score, prediction = verification.verify_files(speaker_filepath, output_file)
        # waveform_x = verification.load_audio(speaker_filepath) # waveform_x.shape: [60930]
        # waveform_y = verification.load_audio(output_file) # waveform_y.shape: [60930]
        # print('waveform_x.shape: {}, waveform_y.shape: {}'.format(waveform_x.shape, waveform_y.shape))
        # # if torch.equal(waveform_x,waveform_y):
        # #     print('{}/{}: output is the same as reference! referece: {} = output: {}'.format(
        # #         i, num_processed, speaker_file, output_file))
        
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

    # get avg. score with down-sampling on paired reference (tetiana: 0.95, mykyta: 0.94, lada: 0.95, dmytro: 0.94, oleksa: 0.94)
    # get avg. score without down-samping on paired reference (tetiana: 0.12, mykyta: 0.46, lada: 0.49, dmytro: 0.27, oleksa: 0.32)
    # get avg. score with down-sampling on unpaired but sorted reference (tetiana: 0.75, mykyta: 0.78, oleksa: 0.80, lada: 0.67)
    score_list = [float(s[0].numpy()) for s in scores[:i]]
    score_mean = np.mean(score_list)
    print('mean similarity score for speaker {}: {:.2f}'.format(voice, score_mean))

    # get prediction accuracy with down-sampling on paired reference (tetiana: 100%, mykyta: 100%, lada: 100%, dmytro: 100%, oleksa: 100%)
    # get prediction accuracy without down-sampling on paired reference (tetiana: 0%, mykyta: 44.44%, lada: 55.56%, dmytro: 66.67%, oleksa: 88.89%)
    # get prediction accuracy with down-sampling on unpaired but sorted reference (tetiana: 100%, mykyta: 100%, oleksa: 100%, lada: 100%)
    prediction_list = [int(p) for p in predictions[:i]]
    prediction_mean = np.mean(prediction_list)
    print('prediction accuracy for speaker {}: {:.2f}%'.format(voice, prediction_mean*100))

    # remove symbolic wav files in the work dir (I don't know how they are created)
    wavfiles = glob.glob(os.path.join(work_path, '*.wav'))
    for f in wavfiles:
        if os.path.isfile(f):
            os.remove(f)

    # # get the avg. duration of the syn files with the scaled paired reference files (5.01 seconds)
    # appendix = 'paired'
    # output_files = glob.glob(os.path.join(output_path, '*_{}.wav'.format(appendix)))
    # dur_output = np.mean([wav_duration(f) for f in output_files])
    # print('avg. duration of the syn files with scaled paired reference files: {:.2f} seconds'.format(dur_output))

    # # get the avg. duration of the syn files with the regular paired reference files (5.91 seconds)
    # output_path0 = output_path.replace('_scaled', '')
    # appendix = 'new'
    # output_files0 = glob.glob(os.path.join(output_path0, '*_{}.wav'.format(appendix)))
    # dur_output0 = np.mean([wav_duration(f) for f in output_files0])
    # print('avg. duration of the syn files with the regular paired reference files: {:.2f} seconds'.format(dur_output0))

    # # get the avg. duration of the regular reference files (3.98 seconds)
    # output_files_ref = glob.glob(os.path.join(output_path0, '*.wav'))
    # output_files_ref = filter_path(speaker_filepaths, keywords=['_new', '16000'])
    # output_files_ref = output_files_ref[:num_processed]
    # dur_output_ref = np.mean([wav_duration(f) for f in output_files_ref])
    # print('avg. duration of the regular reference files: {:.2f} seconds'.format(dur_output_ref))
