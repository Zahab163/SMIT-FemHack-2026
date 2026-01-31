import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load model + preprocessing pipeline
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")          # trained ML model
    pipeline = joblib.load("preprocess.pkl")  # preprocessing pipeline
    return model, pipeline

model, pipeline = load_model()

# -------------------------------
# Risk Label Function
# -------------------------------
def assign_risk_label(score):
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Medium"
    else:
        return "Low"

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎓 Student Dropout Early Warning System")

uploaded_file = st.file_uploader("Upload student CSV file", type=["csv"])

if uploaded_file:
    # Load data
    student_df = pd.read_csv(uploaded_file)

    # Preprocess + Predict
    X = pipeline.transform(student_df)
    probs = model.predict_proba(X)[:, 1]  # dropout probability
    preds = model.predict(X)

    # Build results dataframe
    results = pd.DataFrame({
        "student_id": student_df.index,
        "risk_score": probs,
        "risk_label": [assign_risk_label(s) for s in probs],
        "predicted_dropout": preds
    })

    # -------------------------------
    # Show Top High-Risk Students
    # -------------------------------
    st.subheader("Top 20 High-Risk Students")
    top_risk = results.sort_values("risk_score", ascending=False).head(20)
    st.dataframe(top_risk)

    # -------------------------------
    # Individual Student View
    # -------------------------------
    st.subheader("Check Individual Student Risk")
    student_choice = st.selectbox("Select student_id", results["student_id"])
    student_row = results.loc[results["student_id"] == student_choice]

    st.write("Risk Score:", float(student_row["risk_score"]))
    st.write("Risk Label:", student_row["risk_label"].values[0])
    st.write("Predicted Dropout:", int(student_row["predicted_dropout"]))

    # -------------------------------
    # Feature Importance (Simple)
    # -------------------------------
    st.subheader("Top Reasons (Feature Importance)")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": pipeline.get_feature_names_out(),
            "importance": importances
        }).sort_values("importance", ascending=False).head(10)
        st.bar_chart(feat_imp.set_index("feature"))
    else:
        st.write("Feature importance not available for this model.")