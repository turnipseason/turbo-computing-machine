from vosk import Model, KaldiRecognizer
import json
import os
import requests
import zipfile
import wave
import moviepy.editor as mp
from normalize import normalize_text
import streamlit as st

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

    def transcribe(self, transcription, tokenizer, model):
        if transcription:
            decoded_output = normalize_text(tokenizer, model, transcription)
            st.markdown(f"<span style='font-size:30px'>Нормализованный текст</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:30px'>{decoded_output}</span>", unsafe_allow_html=True)
        else:
            st.write("Модель VOSK не смогла распознать речь в вашем файле!", font_size='20')

    def process_uploaded_file(self, uploaded_file, tokenizer, model):
        audio_file_path = self.get_audio(uploaded_file)
        if audio_file_path:
            transcription = self.process_recording(audio_file_path)
            st.markdown(f"<span style='font-size:30px'>Транскрибация моделью VOSK:</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:30px'>{transcription}</span>", unsafe_allow_html=True)
            self.transcribe(transcription, tokenizer, model)
        else:
            st.write("Неподдерживаемый формат файла", font_size='20')

    def process_recorded_audio(self, audio_data, tokenizer, model):
        self.save_audio(audio_data)
        transcription = self.process_recording("recorded_audio.wav")
        st.write("Транскрибация моделью VOSK:", font_size='20')
        st.write(transcription, font_size='20')
        self.transcribe(transcription, tokenizer, model)