from vosk import Model, KaldiRecognizer
import json
import os
import requests
import zipfile
import wave
import moviepy.editor as mp


class AudioProcessor:
    def __init__(self, vosk_model_path):
        self.vosk_model_path = vosk_model_path
        self.transcription = ""
        self.download_and_extract_vosk_model()
        self.vosk_model = Model(vosk_model_path)

        self.rec = KaldiRecognizer(self.vosk_model, 44100)
        self.rec.SetWords(True)

    def download_and_extract_vosk_model(self):
        url = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
        model_path = self.vosk_model_path

        if not os.path.exists(model_path):
            if not os.path.exists("model"):
                os.makedirs("model")

            response = requests.get(url, stream=True)
            with open("model/vosk-model-small-ru-0.22.zip", "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            with zipfile.ZipFile("model/vosk-model-small-ru-0.22.zip", "r") as zip_ref:
                zip_ref.extractall("model")

    def get_audio(self, uploaded_file):
        file_type = uploaded_file.type
        if file_type in ["audio/mp3", "audio/wav"]:
            audio_file_path = "temp_audio.wav"
            with open(audio_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        elif file_type == "video/mp4":
            video_file_path = "temp_video.mp4"
            with open(video_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            audio_file_path = self.extract_audio_from_video(video_file_path)
        else:
            audio_file_path = None
        return audio_file_path

    def save_audio(self, audio_data):
        with wave.open("recorded_audio.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(audio_data)

    def extract_audio_from_video(self, video_file):
        video = mp.VideoFileClip(video_file)
        audio_file = "temp_audio.wav"
        video.audio.write_audiofile(audio_file)
        return audio_file

    def process_recording(self, file_path):
        wf = wave.open(file_path, "rb")
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if self.rec.AcceptWaveform(data):
                result = json.loads(self.rec.Result())
                self.transcription += result.get('text', '') + ' '

        result = json.loads(self.rec.FinalResult())
        self.transcription += result.get('text', '')

        return self.transcription.strip()
