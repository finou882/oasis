import requests
import json
import pyaudio

def vvox_test(text):
    # Start your voicebox engine.
    host = "localhost"
    port = 50021
    
    params = (
        ('text', text),
        ('speaker', 14),
    )
    

    query = requests.post(
        f'http://{host}:{port}/audio_query',
        params=params
    )

    synthesis = requests.post(
        f'http://{host}:{port}/synthesis',
        headers = {"Content-Type": "application/json"},
        params = params,
        data = json.dumps(query.json())
    )
    

    voice = synthesis.content
    pya = pyaudio.PyAudio()
    
    stream = pya.open(format=pyaudio.paInt16,
                      channels=1,
                      rate=24000,
                      output=True)
    
    stream.write(voice)
    stream.stop_stream()
    stream.close()
    pya.terminate()
    
if __name__ == "__main__":
    text = "こんにちは。"
    vvox_test(text)
