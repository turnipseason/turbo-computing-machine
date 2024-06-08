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


st.set_page_config(
    page_title="LaTeXT5_DEMO",
    page_icon=":four_leaf_clover:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

def loading_spinner(func):
    with st.spinner("Ваш запрос обрабатывается..."):
        func()

st.header("Работа с текстом")
user_input = st.text_area('Введите текст для нормализации:')

if st.button('Нормализовать текст'):
    loading_spinner(lambda: transcriber.transcribe(user_input, tokenizer, model))

st.header("Работа с аудио/видео файлами")
uploaded_file = st.file_uploader("Загрузите аудио или видео файл", type=["mp3", "mp4", "wav"])

if uploaded_file is not None:
    if uploaded_file.type.startswith('audio'):
        st.audio(uploaded_file, format='audio/wav')
    elif uploaded_file.type.startswith('video'):
        st.video(uploaded_file, format='video/mp4')
    if st.button('Нормализовать аудио'):
        loading_spinner(lambda: transcriber.process_uploaded_file(uploaded_file, tokenizer, model))


st.header("Работа с произвольной записью голоса:")
recorded_audio=st_audiorec()

if recorded_audio is not None:
    if st.button('Нормализовать запись голоса'):
        loading_spinner(lambda: transcriber.process_recorded_audio(recorded_audio, tokenizer, model))
