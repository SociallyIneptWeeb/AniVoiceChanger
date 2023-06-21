import sys
import wave
from pathlib import Path
from time import sleep, time
from os import getenv
from urllib.parse import urlencode

import keyboard
from threading import Thread

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import pyaudio
import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv


# load environment variables
load_dotenv()
COLAB_URL = getenv('COLAB_URL')
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
MICROPHONE_ID = int(getenv('MICROPHONE_ID'))
SPEAKERS_INPUT_ID = int(getenv('SPEAKERS_INPUT_ID'))


def rvc_infer_colab():
    params_encoded = urlencode({'model': MODEL_NAME, 'pitch': PITCH_CHANGE, 'algo': PITCH_EXTRACTION_ALGO, 'volume': VOLUME_ENVELOPE, 'index_rate': INDEX_RATE})

    with open(INPUT_VOICE_PATH, 'rb') as infile:
        files = {'audio_file': infile}
        r = requests.post(f'{COLAB_URL}/infer?{params_encoded}', files=files)
    
    with open(OUTPUT_VOICE_PATH, 'wb') as outfile:
        outfile.write(r.content)


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
    rvc_infer_colab()
    print(f'Time taken for RVC voice conversion: {time() - start_time}s')

    # play to both app mic input and speakers
    threads = [Thread(target=play_voice, args=[CABLE_INPUT_ID]), Thread(target=play_voice, args=[SPEAKERS_INPUT_ID])]
    [t.start() for t in threads]
    [t.join() for t in threads]


if __name__ == '__main__':
    INPUT_VOICE_PATH = str(BASE_DIR / 'AniVoiceChanger' / 'audio' / 'input.mp3')
    OUTPUT_VOICE_PATH = str(BASE_DIR / 'AniVoiceChanger' / 'audio' / 'output.wav')
    CHUNK = 1024
    FORMAT = pyaudio.paInt16

    p = pyaudio.PyAudio()

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
