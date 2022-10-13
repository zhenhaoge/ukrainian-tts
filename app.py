import tempfile
import gradio as gr
from datetime import datetime
from enum import Enum
from ukrainian_tts.tts import TTS, Stress, Voices
from torch.cuda import is_available
from os import getenv
from data_logger import log_data
from threading import Thread
from queue import Queue


def check_thread(logging_queue: Queue):
    logging_callback = log_data(hf_token=getenv("HF_API_TOKEN"), dataset_name="uk-tts-output", private=True)
    while True:
        item = logging_queue.get()
        logging_callback(item)

if getenv("HF_API_TOKEN") is not None:
    log_queue = Queue()
    t = Thread(target=check_thread, args=(log_queue,))
    t.start()


class StressOption(Enum):
    AutomaticStress = "Автоматичні наголоси (за словником) 📖"
    AutomaticStressWithModel = "Автоматичні наголоси (за допомогою моделі) 🧮"


class VoiceOption(Enum):
    Olena = "Олена (жіночий) 👩"
    Mykyta = "Микита (чоловічий) 👨"
    Lada = "Лада (жіночий) 👩"
    Dmytro = "Дмитро (чоловічий) 👨"
    Olga = "Ольга (жіночий) 👩"

print(f"CUDA available? {is_available()}")

badge = (
    "https://visitor-badge-reloaded.herokuapp.com/badge?page_id=robinhad.ukrainian-tts"
)

ukr_tts = TTS(use_cuda=is_available())


def tts(text: str, voice: str, stress: str):
    print("============================")
    print("Original text:", text)
    print("Voice", voice)
    print("Stress:", stress)
    print("Time:", datetime.utcnow())

    voice_mapping = {
        VoiceOption.Olena.value: Voices.Olena.value,
        VoiceOption.Mykyta.value: Voices.Mykyta.value,
        VoiceOption.Lada.value: Voices.Lada.value,
        VoiceOption.Dmytro.value: Voices.Dmytro.value,
        VoiceOption.Olga.value: Voices.Olga.value,
    }
    stress_mapping = {
        StressOption.AutomaticStress.value: Stress.Dictionary.value,
        StressOption.AutomaticStressWithModel.value: Stress.Model.value
    }

    speaker_name = voice_mapping[voice]
    stress_selected = stress_mapping[stress]
    text_limit = 7200
    text = (
        text if len(text) < text_limit else text[0:text_limit]
    )  # mitigate crashes on hf space
    
    if getenv("HF_API_TOKEN") is not None:
        log_queue.put([text, speaker_name, stress_selected, str(datetime.utcnow())])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        _, text = ukr_tts.tts(text, speaker_name, stress_selected, fp)
        return fp.name, text


with open("README.md") as file:
    article = file.read()
    article = article[article.find("---\n", 4) + 5::]


iface = gr.Interface(
    fn=tts,
    inputs=[
        gr.components.Textbox(
            label="Input",
            value="Введіть, будь ласка, своє р+ечення.",
        ),
        gr.components.Radio(
            label="Голос",
            choices=[option.value for option in VoiceOption],
            value=VoiceOption.Olena.value,
        ),
        gr.components.Radio(
            label="Наголоси",
            choices=[option.value for option in StressOption],
            value=StressOption.AutomaticStress.value
        ),
    ],
    outputs=[
        gr.components.Audio(label="Output"),
        gr.components.Textbox(label="Наголошений текст"),
    ],
    title="🐸💬🇺🇦 - Coqui TTS",
    description="Україномовний🇺🇦 TTS за допомогою Coqui TTS (щоб вручну поставити наголос, використовуйте + перед голосною)",
    article=article + f'\n  <center><img src="{badge}" alt="visitors badge"/></center>',
    examples=[
        [
            "Введіть, будь ласка, своє речення.",
            VoiceOption.Olena.value,
            StressOption.AutomaticStress.value,
        ],
        [
            "Введіть, будь ласка, своє речення.",
            VoiceOption.Mykyta.value,
            StressOption.AutomaticStress.value,
        ],
        [
            "Вв+едіть, будь ласка, св+оє реч+ення.",
            VoiceOption.Dmytro.value,
            StressOption.AutomaticStress.value,
        ],
        [
            "Привіт, як тебе звати?",
            VoiceOption.Olga.value,
            StressOption.AutomaticStress.value,
        ],
        [
            "Договір підписано 4 квітня 1949 року.",
            VoiceOption.Lada.value,
            StressOption.AutomaticStress.value,
        ],
    ]
)
iface.launch(enable_queue=True)
