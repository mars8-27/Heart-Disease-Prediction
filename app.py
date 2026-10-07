"""Streamlit UI: collects patient info, calls the model, stores to PostgreSQL."""
import streamlit as st

from database import get_history, init_db, save_record
from predict import explain, predict

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")


@st.cache_resource
def setup_database() -> bool:
    """Create tables once per server session. Returns False if DB is unreachable."""
    try:
        init_db()
        return True
    except Exception as exc:  # noqa: BLE001
        st.session_state["db_error"] = str(exc)
        return False


db_ok = setup_database()

st.title("❤️ Heart Disease Prediction System")
st.caption("Educational demo built with Streamlit, XGBoost and PostgreSQL. Not a medical device.")

if not db_ok:
    st.warning("Database unavailable - predictions will work but will not be saved.")

tab_predict, tab_history = st.tabs(["Predict", "History"])

with tab_predict:
    with st.form("patient_form"):
        name = st.text_input("Patient name (optional)")

        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
            sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            cp = st.selectbox(
                "Chest pain type",
                [1, 2, 3, 4],
                format_func=lambda x: {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}[x],
            )
            trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 120)
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

        with c2:
            fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x else "No")
            restecg = st.selectbox(
                "Resting ECG",
                [0, 1, 2],
                format_func=lambda x: ["Normal", "ST-T abnormality", "LV hypertrophy"][x],
            )
            thalach = st.number_input("Max heart rate achieved", 60, 220, 150)
            exang = st.selectbox("Exercise-induced angina", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with c3:
            oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
            slope = st.selectbox(
                "ST slope",
                [1, 2, 3],
                format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x],
            )
            ca = st.selectbox("Major vessels colored by fluoroscopy", [0, 1, 2, 3])
            thal = st.selectbox(
                "Thalassemia",
                [3, 6, 7],
                format_func=lambda x: {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}[x],
            )

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        features = {
            "age": int(age), "sex": int(sex), "cp": int(cp), "trestbps": int(trestbps),
            "chol": int(chol), "fbs": int(fbs), "restecg": int(restecg),
            "thalach": int(thalach), "exang": int(exang), "oldpeak": float(oldpeak),
            "slope": int(slope), "ca": int(ca), "thal": int(thal),
        }

        try:
            prediction, probability = predict(features)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()

        if prediction == 1:
            st.error(f"High risk of heart disease - probability {probability:.1%}")
        else:
            st.success(f"Low risk of heart disease - probability {probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))

        try:
            contrib = explain(features).head(8)
            st.subheader("Why this prediction?")
            st.caption(
                "Top factors (SHAP values). Positive pushes the risk up, negative pushes it down."
            )
            st.bar_chart(contrib)
        except Exception as exc:  # noqa: BLE001
            st.info(f"Explanation unavailable: {exc}")

        if db_ok:
            try:
                record_id = save_record(name, features, prediction, probability)
                st.caption(f"Saved to database (record #{record_id}).")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not save record: {exc}")

with tab_history:
    if not db_ok:
        st.info("Connect PostgreSQL to see saved assessments.")
    else:
        if st.button("Refresh"):
            st.rerun()
        try:
            history = get_history()
            if history.empty:
                st.info("No assessments saved yet.")
            else:
                st.dataframe(history, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load history: {exc}")
