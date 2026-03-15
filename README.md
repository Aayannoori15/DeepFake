# Deep Fake
# website Link:
https://deepfake-3ux1.onrender.com/
Real vs fake image detection using a custom CNN served with Django.

## What This Project Does
- Takes an image upload in the browser.
- Preprocesses it to match training (32x32 RGB, scaled to [0,1]).
- Runs the PyTorch CNN model on CPU.
- Returns a verdict and a "looks fake" score (P(fake)).
- Shows a live preview before submit, and shows the uploaded image in results.
- Keeps uploads **in memory only** (no file/database storage).

## Project Structure
- `deepfake_web/`: Django project settings and URL wiring.
- `predictor/`: Django app with model + inference logic.
- `templates/index.html`: UI.
- `static/predictor/style.css`: Styling.
- `best_model (1).pth`: PyTorch model weights (local only, not committed).

## CNN Architecture
Input: 3x32x32

```
Conv2d(3, 32, 3x3, padding=1) -> ReLU -> MaxPool(2x2)
Conv2d(32, 64, 3x3, padding=1) -> ReLU -> MaxPool(2x2)
Conv2d(64, 128, 3x3, padding=1) -> ReLU -> MaxPool(2x2)
Flatten (4*4*128 = 2048)
Linear(2048 -> 256) -> ReLU
Linear(256 -> 1) -> Sigmoid
```

Output is a single sigmoid probability: **P(fake)**.

## Inference Workflow
1. User uploads image from the UI.
2. Django reads file bytes into memory (no disk write).
3. PIL opens image and converts to RGB.
4. Image is resized to 32x32, scaled to [0,1].
5. Tensor shape becomes `[1, 3, 32, 32]`.
6. Model runs in `eval()` with `torch.no_grad()`.
7. Output is interpreted as:
   - P(fake) >= 0.5 → "This image is probably fake."
   - P(fake) < 0.5 → "This image is probably real."

## Setup
### 1) Create/activate venv
This project expects a venv at:
`/Users/aayannoori/Desktop/django/.venv`

### 2) Install dependencies
```
/Users/aayannoori/Desktop/django/.venv/bin/python -m pip install torch django pillow psycopg[binary]
```

### 3) Run server
```
/Users/aayannoori/Desktop/django/.venv/bin/python manage.py runserver
```

Open: `http://127.0.0.1:8000/`

## Database
PostgreSQL is configured by default with auto-fallback to SQLite if Postgres is unavailable.

Edit database settings in:
`deepfake_web/settings.py`

## Notes
- If you used normalization during training, add the same normalization in
  `predictor/views.py` inside `_preprocess_image`.
- The model file is excluded from git by default.
