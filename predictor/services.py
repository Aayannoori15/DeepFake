from __future__ import annotations

import base64
import io
import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from torch import nn

try:
    from nltk.stem import PorterStemmer
except Exception:  # pragma: no cover - optional dependency fallback
    PorterStemmer = None

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_MODEL_PATH = BASE_DIR / "best_model (1).pth"
TEXT_MODEL_PATH = BASE_DIR / "best_rnn_weights.pth"
TEXT_VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

STOPWORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer() if PorterStemmer else None


class ImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class EssayRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


@lru_cache(maxsize=1)
def get_image_model() -> ImageCNN:
    if not IMAGE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing image weights: {IMAGE_MODEL_PATH}")

    model = ImageCNN()
    state = torch.load(IMAGE_MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_text_assets() -> tuple[EssayRNN, object]:
    if not TEXT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing text weights: {TEXT_MODEL_PATH}")
    if not TEXT_VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Missing text vectorizer. Add tfidf_vectorizer.pkl to enable essay predictions."
        )

    with open(TEXT_VECTORIZER_PATH, "rb") as handle:
        vectorizer = pickle.load(handle)

    input_size = len(vectorizer.get_feature_names_out())
    model = EssayRNN(input_size=input_size)
    state = torch.load(TEXT_MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, vectorizer


def text_assets_ready() -> bool:
    return TEXT_MODEL_PATH.exists() and TEXT_VECTORIZER_PATH.exists()


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((32, 32))
    array = np.array(image).astype("float32") / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    tokens = []
    for token in text.split():
        if not token or token in STOPWORDS:
            continue
        if STEMMER:
            token = STEMMER.stem(token)
        tokens.append(token)
    return " ".join(tokens)


def image_preview_data_url(image_bytes: bytes, content_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{content_type};base64,{encoded}"
