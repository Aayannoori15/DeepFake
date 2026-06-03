# Deep Fake
# website Link:
https://deepfake-3ux1.onrender.com/
Deep Fake is a Django-based AI content detection studio for checking whether an image looks fake and whether an essay looks AI-generated.
Real vs fake image detection using a custom CNN served with Django.

## What It Does
- Image detector powered by a custom PyTorch CNN.
- Text detector powered by a PyTorch RNN with TF-IDF features.
- Shared website shell with a navbar and room for more detectors later.
- Clean, responsive UI with live image preview and separate pages for each tool.
- Volatile uploads only. Nothing is persisted after inference.

## Website Structure
- `Home` introduces the product and links to each detector.
- `Image Check` accepts an image upload and returns a real/fake verdict.
- `Text Check` accepts an essay and returns an AI/human-written verdict.

## Project Architecture
- `deepfake_web/`
  - Django project settings, URLs, and WSGI entry point.
- `predictor/`
  - Views, forms, and model-loading helpers.
- `templates/`
  - `base.html` shared shell.
  - `home.html` landing page.
  - `image_detector.html` image workflow.
  - `text_detector.html` essay workflow.
- `static/predictor/`
  - CSS for the entire site.

## Model Architecture

### Image Model
Input: `3 x 32 x 32`

```text
Conv2d(3 -> 32, kernel=3, padding=1)
ReLU
MaxPool2d(2)

Conv2d(32 -> 64, kernel=3, padding=1)
ReLU
MaxPool2d(2)

Conv2d(64 -> 128, kernel=3, padding=1)
ReLU
MaxPool2d(2)

Flatten
Linear(2048 -> 256)
ReLU
Linear(256 -> 1)
Sigmoid
```

### Text Model
Input: TF-IDF features with `max_features=5000`

```text
TF-IDF vectorizer
RNN(input_size=5000, hidden_size=128)
Linear(128 -> 1)
Sigmoid
```

## Inference Flow
1. User opens the relevant page from the navbar.
2. Django receives the image or text input.
3. The input is normalized to match training.
4. The model runs in evaluation mode with `torch.no_grad()`.
5. The app returns a human-readable verdict and a score.

## Local Setup
### 1) Activate the virtual environment
The project expects the venv at:
`/Users/aayannoori/Desktop/django/.venv`

### 2) Install dependencies
```bash
/Users/aayannoori/Desktop/django/.venv/bin/python -m pip install -r requirements.txt
```

### 3) Run the server
```bash
/Users/aayannoori/Desktop/django/.venv/bin/python manage.py runserver
```

Open:
`http://127.0.0.1:8000/`

## Deployment Notes
- Gunicorn is the production server.
- WhiteNoise serves static files.
- PostgreSQL is preferred, with SQLite fallback if Postgres is unavailable.

## Deploying on Render

- Store secrets (API keys) in Render's Environment Variables dashboard — do NOT
  commit them to the repository or store them in a checked-in `.env` file.
- Recommended variable names:
  - `GROQ_API_KEY` — your Groq/OpenAI-style API key used by the agent.
  - Any additional provider keys required by tools you enable.
- After adding environment variables in Render, trigger a deploy or restart the
  service so the new variables take effect.

For local development, keep secrets in a local `.env` file and make sure
`.env` is included in `.gitignore` (it already is in this repo). The application
loads local variables via `python-dotenv` during development, while production
reads from the environment.

## Memory Optimization for Render

This app uses PyTorch models and an LLM agent, which can consume significant memory.
If you see `Worker exiting (pid:...) was sent SIGKILL! Perhaps out of memory?`:

**Settings already optimized:**
- `DEBUG = False` in production (environment-based).
- Gunicorn configured with 1 worker (not 3+) and periodic restarts.
- Image/text models cached with `@lru_cache` to avoid reloading.
- Input size validation to prevent huge payloads.

**If still hitting memory limits on Render:**
1. Upgrade to a larger plan (Standard has more memory than Starter/Free).
2. Reduce the number of concurrent requests (Render Free tier may support only 1-2).
3. Consider lazy-loading models only on first use per request (tradeoff: slower first response).
4. Reduce `max_tokens` or increase timeouts in the agent if queries are slow.

## Important Files
- [`predictor/services.py`](/Users/aayannoori/Desktop/django/DeepFake/predictor/services.py)
- [`predictor/views.py`](/Users/aayannoori/Desktop/django/DeepFake/predictor/views.py)
- [`templates/base.html`](/Users/aayannoori/Desktop/django/DeepFake/templates/base.html)
- [`templates/image_detector.html`](/Users/aayannoori/Desktop/django/DeepFake/templates/image_detector.html)
- [`templates/text_detector.html`](/Users/aayannoori/Desktop/django/DeepFake/templates/text_detector.html)

## Missing Text Artifact
The essay detector also expects:
- `best_rnn_weights.pth`
- `tfidf_vectorizer.pkl`

The weights file is already in the repo. If the vectorizer is not present yet, the page still renders and shows a setup hint instead of crashing.
