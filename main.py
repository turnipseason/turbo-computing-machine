import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
from normalize import normalize_text
from audio import AudioProcessor
from st_audiorec import st_audiorec

os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "500"

vosk_model_path = "model/vosk-model-small-ru-0.22"
transcriber = AudioProcessor(vosk_model_path)

output_dir = 'turnipseason/latext5'
model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

st.header("Обработка текста")
user_input = st.text_area('Введите текст для нормализации:')

if st.button('Нормализовать текст'):
    if user_input:
        decoded_output = normalize_text(tokenizer, model, user_input)
        st.write('Нормализованный текст:')
        st.write(decoded_output)
    else:
        st.write('Пожалуйста, введите текст.')


st.header("Обработка аудио/видео файлов")
uploaded_file = st.file_uploader("Загрузите аудио или видео файл", type=["mp3", "mp4", "wav"])

if uploaded_file is not None:
    if st.button('Нормализовать аудио'):
        audio_file_path = transcriber.get_audio(uploaded_file)
        if audio_file_path:
            transcription = transcriber.process_recording(audio_file_path)
            st.write("Транскрибация моделью VOSK:")
            st.write(transcription)

            if transcription:
                decoded_output = normalize_text(tokenizer, model, transcription)
                st.write("Нормализованный текст:")
                st.write(decoded_output)
            else:
                st.write("Модель VOSK не смогла распознать речь в вашем файле!")
        else:
            st.write("Неподдерживаемый формат файла")

st.header("Обработка произвольной записи голоса:")

recorded_audio=st_audiorec()

if recorded_audio is not None:
    if st.button('Нормализовать запись голоса'):
        transcriber.save_audio(recorded_audio)
        transcription = transcriber.process_recording("recorded_audio.wav")
        st.write("Транскрибация моделью VOSK:")
        st.write(transcription)

        if transcription:
            decoded_output = normalize_text(tokenizer, model, transcription)
            st.write("Нормализованный текст:")
            st.write(decoded_output)
        else:
            st.write("Модель VOSK не смогла распознать речь в вашем файле!")

