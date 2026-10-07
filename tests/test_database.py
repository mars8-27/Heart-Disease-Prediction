from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import database


def test_patient_can_be_persisted(monkeypatch, patient_features):
    engine = create_engine("sqlite:///:memory:")
    database.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    monkeypatch.setattr(database, "SessionLocal", Session)

    record_id = database.save_record(
        "Test Patient",
        patient_features,
        prediction=1,
        probability=0.82,
    )

    with Session() as session:
        record = session.get(database.Patient, record_id)

    assert record is not None
    assert record.name == "Test Patient"
    assert record.prediction == 1
    assert record.probability == 0.82


def test_patient_name_is_optional(monkeypatch, patient_features):
    engine = create_engine("sqlite:///:memory:")
    database.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    monkeypatch.setattr(database, "SessionLocal", Session)

    record_id = database.save_record(
        "",
        patient_features,
        prediction=0,
        probability=0.18,
    )

    with Session() as session:
        record = session.get(database.Patient, record_id)

    assert record.name is None
