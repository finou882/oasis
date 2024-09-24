Rem/||(
   Japaneese
   5行目のダブルクオーテーションで囲われているパスをエンジンのrun.exeに変更してください。
   English
   Change the path enclosed in double quotes on the 5th line to the engine run.exe.
   ^)
  )
start "C:\Users\finou\AppData\Local\Programs\VOICEVOX\vv-engine\run.exe"
python3 -m pip install --upgrade pip
python3 -m pip install git+https://github.com/huggingface/accelerate
python3 -m pip install datasets
python3 -m pip install transformers
python3 -m pip install torch
python3 -m pip install json
python3 -m pip install requests
python3 -m pip install pyaudio
python main.py