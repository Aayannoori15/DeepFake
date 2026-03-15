from pathlib import Path

import base64
import io
import numpy as np
import torch
from django.shortcuts import render
from PIL import Image
from torch import nn

from .forms import ImageUploadForm

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "best_model (1).pth"


class CNN(nn.Module):
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
        x = self.fc_layers(x)
        return x


def _load_model():
    model = CNN()
    state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


MODEL = _load_model()


def _preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((32, 32))
    array = np.array(image).astype("float32") / 255.0
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor


def index(request):
    result = None
    probability = None
    probability_pct = None
    image_data_url = None

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data["image"]
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            tensor = _preprocess_image(image)
            with torch.no_grad():
                output = MODEL(tensor)
            probability = float(output.item())
            probability_pct = round(probability * 100, 2)
            result = (
                "This image is probably fake."
                if probability >= 0.5
                else "This image is probably real."
            )
            encoded = base64.b64encode(image_bytes).decode("ascii")
            image_data_url = f"data:{image_file.content_type};base64,{encoded}"
    else:
        form = ImageUploadForm()

    context = {
        "form": form,
        "result": result,
        "probability": probability,
        "probability_pct": probability_pct,
        "image_url": image_data_url,
    }
    return render(request, "index.html", context)
