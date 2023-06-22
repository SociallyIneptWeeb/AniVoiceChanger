import sys
import wave
from pathlib import Path
from time import sleep, time
from os import getenv

import keyboard
from threading import Thread

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import torch
import pyaudio
import sounddevice as sd
import soundfile as sf
from multiprocessing import cpu_count
from dotenv import load_dotenv

from vc_infer_pipeline import VC
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from my_utils import load_audio
from fairseq import checkpoint_utils
from scipy.io import wavfile


# load environment variables
load_dotenv()
MODEL_NAME = getenv('MODEL_NAME')
if MODEL_NAME.endswith('.pth'):
    MODEL_NAME = MODEL_NAME[:-4]
PITCH_CHANGE = int(getenv('PITCH_CHANGE'))
VOLUME_ENVELOPE = float(getenv('VOLUME_ENVELOPE'))
INDEX_RATE = float(getenv('INDEX_RATE')) if getenv('INDEX_RATE') else 0
PITCH_EXTRACTION_ALGO = getenv('PITCH_EXTRACTION_ALGO')
GPU_INDEX = getenv('GPU_INDEX')
MIC_RECORD_KEY = getenv('MIC_RECORD_KEY')
INGAME_PUSH_TO_TALK_KEY = getenv('INGAME_PUSH_TO_TALK_KEY')
MICROPHONE_ID = int(getenv('MICROPHONE_ID')) if getenv('MICROPHONE_ID') else None
SPEAKERS_INPUT_ID = int(getenv('SPEAKERS_INPUT_ID')) if getenv('SPEAKERS_INPUT_ID') else None


class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                    ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                    or "P40" in self.gpu_name.upper()
                    or "1060" in self.gpu_name
                    or "1070" in self.gpu_name
                    or "1080" in self.gpu_name
            ):
                print("16 series/10 series P40 forced single precision")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
        else:
            print("No supported N-card found, use CPU for inference")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G memory config
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G memory config
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


def load_hubert():
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(['hubert_base.pt'], suffix='', )
    hubert = models[0]
    hubert = hubert.to(device)

    if is_half:
        hubert = hubert.half()
    else:
        hubert = hubert.float()

    hubert.eval()
    return hubert


def get_vc():
    model_path = BASE_DIR / 'weights' / f'{MODEL_NAME}.pth'
    if not model_path.exists():
        print(f'The model {model_path} does not exist. Please ensure that you have filled in the proper MODEL_NAME in your .env file.')
        raise Exception()

    model_path = str(model_path)
    print(f'loading pth {model_path}')
    cpt = torch.load(model_path, map_location='cpu')
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)

    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc


def rvc_infer():
    logs_dir = BASE_DIR / 'logs' / MODEL_NAME
    index_path = ''
    for file in logs_dir.iterdir():
        if file.suffix == '.index':
            index_path = str(logs_dir / file.name)
            break
    
    # vc single
    audio = load_audio(INPUT_VOICE_PATH, 16000)
    times = [0, 0, 0]
    if_f0 = cpt.get('f0', 1)
    audio_opt = vc.pipeline(hubert_model, net_g, 0, audio, INPUT_VOICE_PATH, times, PITCH_CHANGE, PITCH_EXTRACTION_ALGO, index_path, INDEX_RATE, if_f0, 3, tgt_sr, 0, VOLUME_ENVELOPE, version, 0.33, f0_file=None)
    wavfile.write(OUTPUT_VOICE_PATH, tgt_sr, audio_opt)


def play_voice(device_id):
    data, fs = sf.read(OUTPUT_VOICE_PATH, dtype='float32')

    if INGAME_PUSH_TO_TALK_KEY:
        keyboard.press(INGAME_PUSH_TO_TALK_KEY)

    sd.play(data, fs, device=device_id)
    sd.wait()

    if INGAME_PUSH_TO_TALK_KEY:
        keyboard.release(INGAME_PUSH_TO_TALK_KEY)


def on_press_key(_):
    global frames, recording, stream
    if not recording:
        frames = []
        recording = True
        stream = p.open(format=FORMAT,
                        channels=MIC_CHANNELS,
                        rate=MIC_SAMPLING_RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=MICROPHONE_ID)


def on_release_key(_):
    global recording, stream
    recording = False
    stream.stop_stream()
    stream.close()
    stream = None

    # if key not held down for long enough
    if not frames or len(frames) < 20:
        print('No audio file to transcribe detected. Hold down the key for a longer time.')
        return

    start_time = time()
    # write microphone audio to file
    wf = wave.open(str(INPUT_VOICE_PATH), 'wb')
    wf.setnchannels(MIC_CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(MIC_SAMPLING_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # voice change
    rvc_infer()
    print(f'Time taken for RVC voice conversion: {time() - start_time}s')

    # play to both app mic input and speakers
    threads = [Thread(target=play_voice, args=[CABLE_INPUT_ID]), Thread(target=play_voice, args=[SPEAKERS_INPUT_ID])]
    [t.start() for t in threads]
    [t.join() for t in threads]


if __name__ == '__main__':
    device = f'cuda:{GPU_INDEX}'
    is_half = True
    config = Config(device, is_half)
    INPUT_VOICE_PATH = str(BASE_DIR / 'AniVoiceChanger' / 'audio' / 'input.mp3')
    OUTPUT_VOICE_PATH = str(BASE_DIR / 'AniVoiceChanger' / 'audio' / 'output.wav')
    CHUNK = 1024
    FORMAT = pyaudio.paInt16

    p = pyaudio.PyAudio()
    if MICROPHONE_ID is None:
        MICROPHONE_ID = p.get_default_input_device_info()['index']

    if SPEAKERS_INPUT_ID is None:
        SPEAKERS_INPUT_ID = p.get_default_output_device_info()['index']

    CABLE_INPUT_ID = None
    for audio_device in sd.query_devices():
        if 'CABLE Input' in audio_device['name']:
            CABLE_INPUT_ID = audio_device['index']
            break

    if not CABLE_INPUT_ID:
        print('Virtual audio cable was not found. Please download and install it.')
        sys.exit()

    # get channels and sampling rate of mic
    mic_info = p.get_device_info_by_index(MICROPHONE_ID)
    MIC_CHANNELS = mic_info['maxInputChannels']
    MIC_SAMPLING_RATE = 40000

    # load hubert model
    hubert_model = load_hubert()

    # get vc
    cpt, version, net_g, tgt_sr, vc = get_vc()

    frames = []
    recording = False
    stream = None

    keyboard.on_press_key(MIC_RECORD_KEY, on_press_key)
    keyboard.on_release_key(MIC_RECORD_KEY, on_release_key)

    try:
        print('Starting voice changer.')
        while True:
            if recording and stream:
                data = stream.read(CHUNK)
                frames.append(data)
            else:
                sleep(0.2)

    except KeyboardInterrupt:
        print('Closing voice changer.')
