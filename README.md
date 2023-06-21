# AniVoiceChanger

An "extension" for Retrieval-based Voice Conversion WebUI. Provides a way to record your voice, convert it using an already trained voice model, and output it in voice-chat of any application without running the webui.


## Setup

### Prerequisites

#### Install 7-Zip

Download and Install the 7-Zip application from [here](https://www.7-zip.org/download.html).

This is used to extract the zipped RVC WebUI application after it has been downloaded.

#### Install Virtual Audio Cable

Download and Install VB-CABLE Driver from [here](https://vb-audio.com/Cable/) by extracting all files and Run Setup Program in administrator mode. Reboot after installation.

This is used to pipe the converted voice audio into the audio input of apps.


### Install RVC WebUI

If you haven't installed the RVC WebUI, download the RVC-beta.7z file from [here](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/RVC-beta.7z) and extract it using 7-Zip into a folder of your choosing.


### Clone AniVoiceChanger repository

Within the extracted RVC-beta folder (Should have a bunch of folders and files), open a command line window and run this command to clone this entire repository.

```git clone https://github.com/SociallyIneptWeeb/AniVoiceChanger```

Run the following command to install the additional python dependencies required for this extension.

```runtime\python.exe -m pip install -r AniVoiceChanger\extra_requirements.txt```


### Filling in your Environment Variables in the .env file

Follow the instructions written in the .env file and fill in the appropriate values. If unsure, please refer to this section of the setup video.


## Usage

This program assumes that you have already trained a voice model, with the model file in the weights folder. If you have not done so, please refer to this section of the setup video.

There are 2 ways to run this program, either locally or using google colab. If you have around 5 GB of GPU VRAM to spare, feel free to run this locally while using the `crepe` pitch extraction algorithm. If you only have around 3 GB of GPU VRAM, you can also run this locally while using the `pm` pitch extraction algorithm. If none of these requirements are met, you should run this using Google Colab.

### Local

To start the program, open a command line window in the extracted RVC-beta folder (Should have a bunch of folders and files) and run this command.

```runtime\python.exe AniVoiceChanger\main_local.py```

Now, hold the RECORD_KEY as defined in your .env file on your keyboard and speak into your mic. For the first time, this might take around 5 seconds to generate and play the voice. For consecutive uses, the time taken will be drastically reduced with caching. Do note that the voice is played into the Cable Output audio device, so if you want to hear it yourself, you may need to use OBS/Audacity to monitor and playback the voice to your speakers. The generated voice will also be written into [this folder](audio/) as `output.wav` file.

### Google Colab

(TODO)

## Terms of Use

The use of the converted voice for the following purposes is prohibited.

* Criticizing or attacking individuals.

* Advocating for or opposing specific political positions, religions, or ideologies.

* Publicly displaying strongly stimulating expressions without proper zoning.

* Selling of voice models and generated voice clips.

* Impersonation of the original owner of the voice with malicious intentions to harm/hurt others.

* Fraudulent purposes that lead to identity theft, fraudulent phone calls and phishing emails.

## Disclaimer

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.
