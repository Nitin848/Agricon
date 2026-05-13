# Agricon

Web application for **crop recommendation**, **fertilizer guidance**, and **plant leaf disease identification**. Built with Flask, scikit-learn, and TensorFlow (disease routes load ML models only when that feature is used).

---

## Features

| Feature | Description |
|---------|-------------|
| **Crop recommendation** | Predicts a suitable crop from soil inputs (N, P, K, pH, rainfall) plus optional city weather via OpenWeatherMap. |
| **Fertilizer recommendation** | Compares user soil nutrients to ideal values per crop and suggests corrective actions. |
| **Disease prediction** | Upload a leaf image; identifies plant type and health status when `.h5` models are present under `Data/`. |

---

## Requirements

- Python **3.11** (see `runtime.txt`; 3.12 often works locally)
- Dependencies listed in `requirements.txt` (includes Flask, pandas, scikit-learn, OpenCV headless, TensorFlow, Gunicorn for Linux hosting)

---

## Local setup

From this folder (`Agricon-main`):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### Crop model (`models/`)

The app loads the **first available** file in this order:

1. `RandomForest.pkl`
2. `DecisionTree.pkl`
3. `NBClassifier.pkl`

Override with:

```bash
set AGRICON_CROP_MODEL=NBClassifier.pkl
```

To regenerate `RandomForest.pkl` from `Data/crop_recommendation.csv`:

```bash
python train_crop_model.py
```

### Disease models (`Data/`)

Place Keras `.h5` files here for full disease prediction (names expected by `app.py`), for example:

- `plant_status_model1.h5`
- `Plant_Leaf_identification_model1.h5`
- Per-crop models such as `Apple_model1.h5`, `tomato_model1.h5`, etc.

If these files are missing, the site still runs; the disease page explains that models are not installed.

### Weather API (optional)

For live temperature and humidity by city, set your API key:

```bash
set OPENWEATHER_API_KEY=your_key_here
```

If the key is missing or the request fails, defaults are used so crop prediction still works.

---

## Run locally

```bash
python app.py
```

- App URL: `http://127.0.0.1:5000` (or `http://localhost:5000`)
- Port can be overridden with `PORT` (e.g. `set PORT=8080`)
- Debug mode: `set FLASK_DEBUG=1`

**Note:** `gunicorn` is for Linux production (see `Procfile`). On Windows, use `python app.py`.

---

## Health check

Useful after deployment:

```http
GET /health
```

Example response:

```json
{"status": "ok", "crop_model_loaded": true}
```

---

## Deploy (example: Render)

1. Push this repository to GitHub.
2. Create a **Web Service** and set **Root Directory** to **`Agricon-main`** if the repo contains the parent `Agricon` folder.
3. Build and start commands are inferred from `requirements.txt`, `runtime.txt`, and `Procfile` (`gunicorn`).
4. Add environment variables as needed (`OPENWEATHER_API_KEY`, optional `AGRICON_CROP_MODEL`).

TensorFlow is memory-heavy; free tiers may fail to build or boot. Use an instance with sufficient RAM if installs or first requests time out.

---

## Project layout (main items)

```
Agricon-main/
├── app.py                 # Flask application
├── config.py              # Default config (e.g. weather key placeholder)
├── requirements.txt
├── Procfile               # Production server (Gunicorn)
├── runtime.txt            # Python version hint for hosts
├── train_crop_model.py    # Train RandomForest → models/RandomForest.pkl
├── Data/                  # CSV datasets + optional .h5 models
├── models/                # Crop recommendation .pkl files
├── templates/             # HTML templates
├── static/                # CSS, scripts
└── uploads/               # Temporary uploads (gitignored)
```

---

## License / attribution

Use and extend according to your team’s policy. Third-party data and model weights should be cited where required.
