import os
import json
from typing import Dict, Any
from .predict import IntentPredictor

# Get absolute path relative to this file's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "banking77")
LABEL_NAMES_PATH = os.path.join(DATA_DIR, "label_names.json")
RESPONSES_PATH = os.path.join(DATA_DIR, "responses.json")


def _load_label_names():
    try:
        with open(LABEL_NAMES_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load label_names.json: {e}")
        return None


def _load_responses():
    # responses.json может содержать маппинг: intent_name -> текст ответа
    try:
        with open(RESPONSES_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load responses.json: {e}")
        return {}


def reply(user_text: str) -> Dict[str, Any]:
    label_names = _load_label_names()
    responses = _load_responses()
    predictor = IntentPredictor(label_names=label_names)
    pred = predictor.predict_intent(user_text, top_k=3)

    top = pred["top"][0]
    intent_name = top["label"]
    confidence = top["prob"]

    answer = responses.get(intent_name)
    # Check if answer exists and is non-empty
    if confidence < 0.1 or answer is None or (isinstance(answer, str) and answer.strip() == ""):
        # Fallback: универсальный ответ, укажем распознанный интент и уверенность
        answer = (
            "Sorry i cant help you with an answer, can you perephrase the question?."
            f"I think your intent is '{intent_name}' (p={confidence:.2f}), but im not sure. "
            "Please provide more details so I can assist you better."
        )

    return {
        "input": user_text,
        "intent": intent_name,
        "confidence": confidence,
        "answer": answer,
        "alternatives": pred["top"],
    }


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "When will my card arrive?"
    result = reply(text)
    print(result)
