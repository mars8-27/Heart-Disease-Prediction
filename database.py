"""Database layer: connects to PostgreSQL and stores patient records."""
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/heart_db",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Patient(Base):
    """One row = one patient assessment (inputs + model output)."""

    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=True)
    age = Column(Integer, nullable=False)
    sex = Column(Integer, nullable=False)
    cp = Column(Integer, nullable=False)
    trestbps = Column(Integer, nullable=False)
    chol = Column(Integer, nullable=False)
    fbs = Column(Integer, nullable=False)
    restecg = Column(Integer, nullable=False)
    thalach = Column(Integer, nullable=False)
    exang = Column(Integer, nullable=False)
    oldpeak = Column(Float, nullable=False)
    slope = Column(Integer, nullable=False)
    ca = Column(Integer, nullable=False)
    thal = Column(Integer, nullable=False)
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    """Create tables if they do not exist yet."""
    Base.metadata.create_all(bind=engine)


def save_record(name: str, features: dict, prediction: int, probability: float) -> int:
    """Insert a patient + result and return the new row id."""
    with SessionLocal() as session:
        record = Patient(
            name=name or None,
            prediction=int(prediction),
            probability=float(probability),
            **features,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id


def get_history(limit: int = 100) -> pd.DataFrame:
    """Return the latest saved assessments as a DataFrame."""
    query = f"SELECT * FROM patients ORDER BY created_at DESC LIMIT {int(limit)}"
    return pd.read_sql(query, engine)
