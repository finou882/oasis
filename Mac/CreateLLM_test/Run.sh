@echo off
start "" "C:\Users\finou\AppData\Local\Programs\VOICEVOX\vv-engine\run.exe"
python
pip install --upgrade pip
pip install git+https://github.com/huggingface/accelerate
pip install datasets
pip install transformers
pip install torch
pip install json
pip install requests
pip install pyaudio
exit
python main.py