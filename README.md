# ❤️ Heart Disease Prediction System

An end-to-end ML web app: enter patient data, get a heart disease risk prediction with probability, and store every assessment in PostgreSQL.

**Stack:** Python · Streamlit · XGBoost · PostgreSQL · SQLAlchemy · Docker

## Architecture

```
User enters patient information
            │
            ▼
       Streamlit UI  (app.py)
            │
            ▼
      Python Backend  (predict.py, database.py)
            │
     ┌──────┴──────┐
     ▼             ▼
PostgreSQL      XGBoost model
 (database.py)  (train_model.py → models/xgb_model.joblib)
     │             │
     ▼             ▼
Store patient   Prediction + probability
information         │
     └──────┬───────┘
            ▼
   Result displayed in Streamlit
```

## Project structure

```
heart-disease-prediction/
├── app.py              # Streamlit UI
├── predict.py          # loads model, returns prediction + probability
├── database.py         # SQLAlchemy model + PostgreSQL helpers
├── prepare_data.py     # downloads + cleans the UCI Heart Disease dataset
├── train_model.py      # trains and saves the XGBoost model
├── Dockerfile          # container image for the Streamlit app
├── docker-compose.yml  # PostgreSQL (+ optional app container)
├── requirements.txt
├── .env.example
├── data/               # heart.csv is generated here
└── models/             # trained model is saved here
```

## Setup (VS Code)

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Get the dataset** (UCI Heart Disease, Cleveland, 303 rows):
   ```bash
   python prepare_data.py
   ```
   This downloads the data with `ucimlrepo`, converts the 0-4 diagnosis into a binary target (0 = no disease, 1 = disease), drops the 6 rows with missing values, and writes `data/heart.csv`.
3. **Start PostgreSQL**
   ```bash
   docker compose up -d
   ```
   Then copy `.env.example` to `.env`.
4. **Train the model**
   ```bash
   python train_model.py
   ```
5. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Run everything in Docker

After `python prepare_data.py` and `python train_model.py` (so `models/xgb_model.joblib` exists):

```bash
docker compose up --build
```

Open http://localhost:8501. The app container reaches Postgres at host `db`. To run only the database and start Streamlit locally, use `docker compose up -d db`.

## Explainability

After each prediction the app shows a SHAP bar chart of the top factors that pushed the risk up or down for that patient (log-odds scale).

## Deploy

- **Streamlit Community Cloud** (free): push to GitHub, connect the repo, and add `DATABASE_URL` in the app's Secrets. Use a hosted Postgres such as Neon or Supabase. Commit the trained model or retrain on startup, since `.gitignore` excludes `models/*.joblib`.
- **Docker / Render / Railway**: containerize the app and point `DATABASE_URL` at a managed Postgres.

## Dataset

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X (CC BY 4.0).

Encoding used by the app (follows UCI): `cp` 1-4, `slope` 1-3, `thal` 3/6/7, `ca` 0-3.

## Disclaimer

Educational project only. Not a medical device and not a substitute for professional diagnosis.
