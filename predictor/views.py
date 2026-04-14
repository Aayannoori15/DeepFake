from io import BytesIO

import torch
from django.shortcuts import render
from PIL import Image

from .forms import ImageUploadForm, TextAnalysisForm
from .services import (
    get_image_model,
    get_text_assets,
    image_preview_data_url,
    preprocess_image,
    preprocess_text,
    text_assets_ready,
)


def home(request):
    return render(
        request,
        "home.html",
        {
            "image_active": False,
            "text_active": False,
        },
    )


def image_detector(request):
    result = None
    probability_pct = None
    image_url = None
    error = None

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data["image"]
            image_bytes = image_file.read()
            image = Image.open(BytesIO(image_bytes))
            tensor = preprocess_image(image)

            try:
                model = get_image_model()
                with torch.no_grad():
                    output = model(tensor)
                probability = float(output.item())
                probability_pct = round(probability * 100, 2)
                # Assumes the positive class in training maps to "fake".
                result = (
                    "This image is probably fake."
                    if probability >= 0.5
                    else "This image is probably real."
                )
                image_url = image_preview_data_url(image_bytes, image_file.content_type)
            except Exception as exc:
                error = str(exc)
    else:
        form = ImageUploadForm()

    return render(
        request,
        "image_detector.html",
        {
            "form": form,
            "result": result,
            "probability_pct": probability_pct,
            "image_url": image_url,
            "error": error,
            "image_active": True,
            "text_active": False,
        },
    )


def text_detector(request):
    result = None
    probability_pct = None
    cleaned_text = None
    error = None
    model_ready = text_assets_ready()

    if request.method == "POST":
        form = TextAnalysisForm(request.POST)
        if form.is_valid():
            raw_text = form.cleaned_data["text"]
            cleaned_text = preprocess_text(raw_text)

            try:
                model, vectorizer = get_text_assets()
                features = vectorizer.transform([cleaned_text]).toarray()
                tensor = torch.from_numpy(features).float().unsqueeze(1)

                with torch.no_grad():
                    output = model(tensor)

                probability = float(torch.sigmoid(output).item())
                probability_pct = round(probability * 100, 2)
                # Flip these labels if your training target encoding was reversed.
                result = (
                    "This essay looks AI-generated."
                    if probability >= 0.5
                    else "This essay looks human-written."
                )
            except FileNotFoundError as exc:
                model_ready = False
                error = str(exc)
            except Exception as exc:
                error = str(exc)
    else:
        form = TextAnalysisForm()

    return render(
        request,
        "text_detector.html",
        {
            "form": form,
            "result": result,
            "probability_pct": probability_pct,
            "cleaned_text": cleaned_text,
            "error": error,
            "model_ready": model_ready,
            "image_active": False,
            "text_active": True,
        },
    )
