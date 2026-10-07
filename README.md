# ❤️ Heart Disease Prediction System

> An end-to-end machine learning application that estimates heart-disease risk from clinical features, explains individual predictions with SHAP, and persists assessments in PostgreSQL.

[![CI](https://github.com/mars8-27/Heart-Disease-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/mars8-27/Heart-Disease-Prediction/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-316192.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED.svg)](https://www.docker.com/)

## Overview

This project demonstrates a complete ML application workflow rather than a standalone notebook:

**Data acquisition → preprocessing → model training → evaluation → serialized model → inference → explainability → database persistence → web UI → containerization → CI**

The application uses the UCI Heart Disease dataset and an XGBoost classifier. Users enter clinical attributes through Streamlit, receive a predicted class and probability, inspect SHAP feature contributions, and optionally review previously saved assessments from PostgreSQL.

> **Important:** This is an educational software project. It is **not a medical device**, diagnostic system, or substitute for professional medical advice.

## ✨ Features

- Interactive Streamlit prediction interface
- XGBoost binary classification
- Probability-based risk output
- SHAP-based local explanation of predictions
- PostgreSQL persistence through SQLAlchemy
- Assessment history view
- Reproducible data preparation and model-training scripts
- Docker + Docker Compose development environment
- Environment-variable based database configuration
- GitHub Actions CI for Python compilation checks
- Clear separation between UI, inference, training, data preparation, and persistence

## 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │   Streamlit Web UI   │
                         │       app.py         │
                         └──────────┬───────────┘
                                    │
                         patient features
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Inference Layer   │
                         │      predict.py      │
                         └───────┬───────┬──────┘
                                 │       │
                         prediction    SHAP
                                 │       │
                    ┌────────────┘       └────────────┐
                    ▼                                 ▼
          ┌──────────────────┐              ┌─────────────────┐
          │ XGBoost Model    │              │ Explanation     │
          │ joblib artifact  │              │ Top contributors│
          └──────────────────┘              └─────────────────┘
                    │
                    │ assessment + result
                    ▼
          ┌──────────────────┐
          │ PostgreSQL       │
          │ database.py      │
          └──────────────────┘


Training pipeline
─────────────────────────────────────────────────────────────

UCI Dataset → prepare_data.py → data/heart.csv
                                  │
                                  ▼
                           train_model.py
                                  │
                                  ▼
                       XGBoost + evaluation
                                  │
                                  ▼
                    models/xgb_model.joblib
```

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| ML | XGBoost, scikit-learn |
| Data | Pandas, NumPy |
| Explainability | SHAP |
| Persistence | PostgreSQL, SQLAlchemy |
| Configuration | python-dotenv |
| Packaging | pip + virtual environments |
| Containers | Docker, Docker Compose |
| CI | GitHub Actions |
| Dataset | UCI Heart Disease |

## 📁 Project Structure

```text
Heart-Disease-Prediction/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI checks
├── app.py                      # Streamlit application
├── predict.py                  # Model loading, inference and SHAP
├── database.py                 # SQLAlchemy models and persistence
├── prepare_data.py             # Dataset download and preprocessing
├── train_model.py              # XGBoost training and evaluation
├── Dockerfile                  # Container image for the app
├── docker-compose.yml          # PostgreSQL + app services
├── .dockerignore               # Docker build exclusions
├── .env.example                # Database configuration template
├── .gitignore                  # Local/generated file exclusions
├── requirements.txt            # Python dependencies
├── data/                       # Generated dataset (not committed)
├── models/                     # Generated model artifact (not committed)
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git
- Docker Desktop
- A local PostgreSQL instance **or** Docker Compose

### 1. Clone the repository

```bash
git clone https://github.com/mars8-27/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset

```bash
python prepare_data.py
```

This retrieves the UCI Heart Disease dataset, converts the original multiclass target into a binary target, removes rows with missing values, and writes `data/heart.csv`.

### 5. Train the model

```bash
python train_model.py
```

The training script:

- removes duplicate rows
- performs a stratified train/test split
- runs 5-fold cross-validation using ROC-AUC
- trains an XGBoost classifier
- reports test accuracy, ROC-AUC and a classification report
- saves the trained model to `models/xgb_model.joblib`

### 6. Configure PostgreSQL

Copy the example environment file:

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

The default configuration targets:

```text
postgresql+psycopg2://postgres:postgres@localhost:5432/heart_db
```

Start PostgreSQL with Docker Compose:

```bash
docker compose up -d db
```

### 7. Run the application

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## 🐳 Run with Docker Compose

The repository includes a PostgreSQL service and an optional Streamlit app service.

First generate the dataset and model:

```bash
python prepare_data.py
python train_model.py
```

Then start the complete stack:

```bash
docker compose up --build
```

The application will be available at **http://localhost:8501**.

To stop the stack:

```bash
docker compose down
```

To remove the PostgreSQL volume as well:

```bash
docker compose down -v
```

> The trained model is intentionally excluded from Git. The Docker Compose setup mounts the local `models/` directory into the app container.

## 🧠 Machine Learning Pipeline

### Input features

The model uses the following 13 clinical features:

```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
```

### Target

The original UCI `num` target is converted into:

- `0` → no heart disease
- `1` → heart disease present

### Model

The current implementation uses `XGBClassifier` with a fixed random seed and explicit hyperparameters for reproducibility.

Evaluation includes:

- 5-fold cross-validation ROC-AUC on the training split
- test accuracy
- test ROC-AUC
- classification report

The README intentionally does not hard-code model scores because they should be regenerated from the current dataset and training configuration.

## 🔎 Explainability

For each prediction, the application uses **SHAP TreeExplainer** to calculate local feature contributions.

Interpretation:

- **Positive SHAP value** → pushes the prediction toward heart disease
- **Negative SHAP value** → pushes the prediction away from heart disease
- Larger absolute values indicate greater contribution to the individual prediction

This makes the application more transparent than exposing only a binary prediction.

## 🗄️ Data Persistence

Each saved assessment can contain:

- optional patient name
- 13 model input features
- predicted class
- predicted probability
- creation timestamp

SQLAlchemy handles the database model and PostgreSQL stores the assessment history.

The application is designed to continue making predictions if PostgreSQL is unavailable, while clearly warning that the assessment cannot be persisted.

## 🔐 Configuration & Security

Environment-specific configuration belongs in `.env`.

Never commit:

- database passwords
- API keys
- production credentials
- `.env` files
- generated model artifacts
- generated datasets containing sensitive information

The repository's `.gitignore` excludes local environments, secrets, generated datasets, and model artifacts.

For a production deployment, use a managed secrets system and a dedicated database user with least-privilege permissions.

## 🧪 CI

GitHub Actions runs on pushes and pull requests to `main`.

Current checks:

1. Install project dependencies
2. Compile the Python modules to catch syntax/import compilation errors

The CI foundation can be expanded with unit tests, linting, type checking, security scanning, and model validation as the project evolves.

## 🌐 Deployment

The application can be deployed using:

- Streamlit Community Cloud
- Docker-based hosting
- Render
- Railway
- Other platforms capable of running the Docker image and connecting to PostgreSQL

For production, use a managed PostgreSQL instance and configure `DATABASE_URL` through the platform's secret/environment configuration.

## 📊 Dataset

**UCI Heart Disease Dataset**

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease*. UCI Machine Learning Repository.

DOI: https://doi.org/10.24432/C52P4X

The dataset is used under its stated license/terms. Refer to the UCI repository for the authoritative dataset documentation.

## 🛣️ Roadmap

- [x] Data preparation pipeline
- [x] XGBoost training pipeline
- [x] Cross-validation and evaluation
- [x] Streamlit interface
- [x] PostgreSQL persistence
- [x] SHAP explainability
- [x] Docker Compose environment
- [x] GitHub Actions CI with automated tests
- [x] Automated unit and integration tests
- [ ] API layer with FastAPI
- [ ] Structured application logging
- [ ] Model/version metadata tracking
- [ ] Automated model validation in CI
- [ ] Production observability
- [ ] Secure authentication and role-based access

## 🤝 Contributing

Contributions are welcome.

A typical workflow:

```bash
git checkout -b feature/your-feature
git add .
git commit -m "Add your feature"
git push origin feature/your-feature
```

Then open a pull request against `main`.

For meaningful changes, include a clear description of the problem, implementation, testing performed, and any trade-offs.

## 📄 License

This project is released under the **Unlicense**. See [LICENSE](LICENSE).

## ⚠️ Disclaimer

This project is intended for **education and software engineering demonstration only**. Predictions are not medical diagnoses and should not be used to make healthcare decisions. No clinical validation or regulatory approval is claimed.
