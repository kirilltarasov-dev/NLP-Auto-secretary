import streamlit as st
import sys
import os

# ==================== ДОБАВЛЕНИЕ ПУТИ К SRC ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.chatbot_logic import reply

# ==================== НАСТРОЙКА STREAMLIT ====================
st.set_page_config(page_title="Banking Assistant", page_icon="💳")

st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #e0e0e0;
    }
    .user-msg {
        background-color: #1a237e;  /* темно-синий для запроса пользователя */
        color: #ffffff;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        text-align: right;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .bot-msg {
        background-color: #2c2c2c;  /* темно-серый для ответа бота */
        color: #ffffff;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        text-align: left;
        max-width: 70%;
        float: left;
        clear: both;
    }
    .chat-container {
        overflow: auto;
        padding-bottom: 20px;
    }
    .stTextInput>div>div>input {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("How can i help you today?")

# ==================== 🔧 ПЕРЕКЛЮЧАТЕЛЬ ДЛЯ РАЗРАБОТЧИКОВ ====================
show_intent = st.sidebar.toggle("Show predicted intent", value=False)

# ==================== ИСТОРИЯ СООБЩЕНИЙ ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================== ОБРАБОТКА ВВОДА ПОЛЬЗОВАТЕЛЯ ====================
user_input = st.chat_input("Введите сообщение...")

if user_input:
    # Получаем ответ от chatbot_logic
    response = reply(user_input)
    answer_text = response.get("answer", "Sorry i cant help you with an answer, can you perephrase the question?.")

        #  Добавляем интент, если включен dev-режим
    if show_intent:
        intent = response.get("intent", "")
        if intent:
            answer_text += f"\n\n [{intent}]"


    # Сохраняем сообщения в сессии
    st.session_state.messages.append({"role": "user", "text": user_input})
    st.session_state.messages.append({"role": "bot", "text": answer_text})

# ==================== ОТОБРАЖЕНИЕ ЧАТА ====================
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{msg['text']}</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
