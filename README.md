# AniVoiceChanger

![](thumbnail.jpg?raw=true)

An "extension" for Retrieval-based Voice Conversion WebUI. Provides a way to record your voice, convert it using a trained voice model, and output it in voice-chat of any application without running the webui.

Showcase: https://www.youtube.com/watch?v=C-PqTbh0LxY

## Setup

### Prerequisites

#### Install Git

Follow the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install Git on your computer.

#### Install 7-Zip

Download and Install the 7-Zip application from [here](https://www.7-zip.org/download.html).

This is used to extract the zipped RVC WebUI application after it has been downloaded.

#### Install Virtual Audio Cable

Download and Install VB-CABLE Driver from [here](https://vb-audio.com/Cable/) by extracting all files and Run Setup Program in administrator mode. Reboot after installation.

This is used to pipe the converted voice audio into the audio input of apps.


### Install RVC WebUI

If you haven't installed the RVC WebUI, download the RVC-beta.7z file from [here](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/RVC-beta.7z) and extract it using 7-Zip into a folder of your choosing. It will take around 8GB of space, not including any voice models that you may train later on.


### Clone AniVoiceChanger repository

Within the extracted RVC-beta folder (Should have a bunch of folders and files), open a command line window and run this command to clone this entire repository and install the additional dependencies required for this extension.

```
git clone https://github.com/SociallyIneptWeeb/AniVoiceChanger
runtime\python.exe -m pip install -r AniVoiceChanger\extra_requirements.txt
```

### Filling in your Environment Variables in the .env file

Follow the instructions written in the .env file and fill in the appropriate values. If unsure, please refer to this section of the setup video.


## Usage

This program assumes that you have already trained a voice model, with the model file in the weights folder. If you have not done so, please refer to this section of the setup video.

Remember to change the audio input device of the game or application you are using to Cable Output (VB-Audio Virtual Cable).

There are 2 ways to run this program, either locally or using google colab. If you have around 5 GB of GPU VRAM to spare, feel free to run this locally while using the `crepe` pitch extraction algorithm. If you only have around 3 GB of GPU VRAM, you can also run this locally while using the `pm` pitch extraction algorithm. If none of these requirements are met, you should run this using Google Colab.

### Local

To start the program, open a command line window in the extracted RVC-beta folder (Should have a bunch of folders and files) and run this command.

```runtime\python.exe AniVoiceChanger\main_local.py```
> Do note that every time a variable is updated in the `.env` file, you will have to rerun this command for the changes to take into effect. E.g. when changing the model name.

Now, hold the RECORD_KEY as defined in your .env file on your keyboard and speak into your mic. For the first time, this might take around 5 seconds to generate and play the voice. For consecutive uses, the time taken will be drastically reduced with caching. The voice will be played into the Cable Output audio device and your speakers as defined in the `.env` file. The generated voice will also be written into [this folder](audio/) as `output.wav` file. 

### Google Colab

Go to [AniVoiceChanger_colab.ipynb](AniVoiceChanger_colab.ipynb) file in Github and click on `Open in Colab` badge. This will open a Colab notebook. Follow the instructions in the notebook to either train a voice model, or run the RVC Inference server.

If you have already uploaded a trained voice model to the Colab runtime and it has started running the inference server, the output of the last cell should display a Ngrok public url. Copy and paste this url into the `COLAB_URL` environment variable in your `.env` file. After all your environment variables are properly set, open a command line window in the extracted RVC-beta folder (Should have a bunch of folders and files) and run this command. Do note that every time a variable is updated in the `.env` file, you will have to rerun this command for the changes to take into effect. E.g. when changing the model name.

```runtime\python.exe AniVoiceChanger\main_colab.py```
> Do note that every time a variable is updated in the `.env` file, you will have to rerun this command for the changes to take into effect. E.g. when changing the model name.

Now, hold the RECORD_KEY as defined in your .env file on your keyboard and speak into your mic. For the first time, this might take around 10 seconds to generate and play the voice. For consecutive uses, the time taken will be drastically reduced with caching. The voice will be played into the Cable Output audio device and your speakers as defined in the `.env` file. The generated voice will also be written into [this folder](audio/) as `output.wav` file.

## Terms of Use

The use of the converted voice for the following purposes is prohibited.

* Criticizing or attacking individuals.

* Advocating for or opposing specific political positions, religions, or ideologies.

* Publicly displaying strongly stimulating expressions without proper zoning.

* Selling of voice models and generated voice clips.

* Impersonation of the original owner of the voice with malicious intentions to harm/hurt others.

* Fraudulent purposes that lead to identity theft or fraudulent phone calls.

## Disclaimer

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.
