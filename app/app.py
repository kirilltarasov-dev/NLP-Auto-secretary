import streamlit as st
import joblib
import os
import json
import numpy as np

# ==================== ПУТИ К ФАЙЛАМ ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "data", "banking77", "logreg_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "data", "banking77", "tfidf_vectorizer.joblib")
LABELS_PATH = os.path.join(BASE_DIR, "..", "data", "banking77", "label_names.json")

# ==================== ЗАГРУЗКА МОДЕЛИ ====================
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

# ==================== НАСТРОЙКА STREAMLIT ====================
st.set_page_config(page_title="Банковский ассистент", page_icon="💳")

st.markdown(
    """
    <style>
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
</style>
    """,
    unsafe_allow_html=True
)

st.title("💬 Ассистент для банковских услуг")

# ==================== СООБЩЕНИЯ ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_bot_response(text: str) -> str:
    X = vectorizer.transform([text])
    pred = model.predict(X)
    # Получаем числовой индекс предсказания корректно
    try:
        pred_idx = int(pred[0]) if hasattr(pred, "__iter__") else int(pred)
    except Exception:
        # на случай, если модель возвращает строку метки
        try:
            return f"📂 Категория вашего запроса: **{str(pred)}**"
        except Exception:
            return "Ошибка при получении предсказания."

    if 0 <= pred_idx < len(labels):
        label = labels[pred_idx]
    else:
        label = str(pred_idx)
    return f"📂 Категория вашего запроса: **{label}**"

user_input = st.chat_input("Введите сообщение...")

if user_input:
    st.session_state.messages.append({"role": "user", "text": user_input})
    bot_response = get_bot_response(user_input)
    st.session_state.messages.append({"role": "bot", "text": bot_response})

# Контейнер для чата
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{msg['text']}</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
