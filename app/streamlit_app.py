from database.crud import (
    save_record,
    get_records,
    clear_records,
    get_statistics
)
import os
import pandas as pd
import streamlit as st
from PIL import Image

from utils.preprocess import preprocess_pil_image
from utils.predict import predict
from utils.gradcam import generate_heatmap, overlay_heatmap
from utils.llm import explain_prediction
from utils.pdf_report import generate_pdf

from database.crud import save_record, get_records, clear_records
os.makedirs("temp", exist_ok=True)
st.set_page_config(
    page_title="AI Medical Intelligence Platform",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Advanced AI Medical Intelligence Platform")
st.markdown("---")

# ==========================
# Patient Information
# ==========================

st.subheader("👤 Patient Information")

name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

st.markdown("---")

left, right = st.columns([1, 1])

# ==========================
# LEFT SIDE
# ==========================

with left:

    uploaded = st.file_uploader(
        "Upload Skin Lesion Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:

        image = Image.open(uploaded).convert("RGB")
        uploaded_image_path = "temp/uploaded_image.jpg"
        image.save(uploaded_image_path)

        st.image(
            image,
            caption="Uploaded Image",
            width="stretch"
        )

# ==========================
# RIGHT SIDE
# ==========================

with right:

    st.subheader("Prediction")

    if uploaded is not None:

        # ------------------------
        # Preprocess
        # ------------------------

        processed = preprocess_pil_image(image)

        # ------------------------
        # Prediction
        # ------------------------

        result = predict(processed)

        st.success(f"Prediction: {result['prediction']}")

        st.metric(
            "Confidence",
            f"{result['confidence']*100:.2f}%"
        )

        # ------------------------
        # Probability Chart
        # ------------------------

        st.subheader("Disease Probabilities")

        prob_df = pd.DataFrame(
            result["probabilities"].items(),
            columns=["Disease", "Probability"]
        ).sort_values(
            "Probability",
            ascending=False
        )

        st.bar_chart(
            prob_df.set_index("Disease")
        )

        # ------------------------
        # Grad-CAM
        # ------------------------

        heatmap = generate_heatmap(
            result["model"],
            processed
        )

        overlay = overlay_heatmap(
            image,
            heatmap
        )
        from PIL import Image
        gradcam_image_path = "temp/gradcam.jpg"
        overlay_pil = Image.fromarray(overlay.astype("uint8")) 
        overlay_pil.save(gradcam_image_path)

        st.subheader("🔥 Explainable AI (Grad-CAM)")

        st.image(
            overlay,
            caption="Regions influencing the prediction",
            width="stretch"
        )
        # ------------------------
        # AI Medical Assistant (Groq)
        # ------------------------

        st.subheader("🤖 AI Medical Assistant")

        with st.spinner("Generating explanation..."):

            explanation = explain_prediction(
                result["prediction"],
                result["confidence"] * 100
            )

        st.markdown(explanation)

        patient = {
            "name": name,
            "age": age,
            "gender": gender
            }
        pdf_path = "temp/medical_report.pdf"
        generate_pdf(
            filename=pdf_path,
            patient=patient,
            result=result,
            explanation=explanation,
            uploaded_image_path=uploaded_image_path,
            gradcam_image_path=gradcam_image_path
            )
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Medical Report (PDF)",
                data=pdf_file,
                file_name="medical_report.pdf",
                mime="application/pdf"
                )

        # ------------------------
        # Save Record
        # ------------------------

        if st.button("💾 Save Record"):

            save_record(
                name,
                age,
                gender,
                result["prediction"],
                result["confidence"]
            )

            st.success("Patient record saved successfully!")

st.markdown("---")

st.header("📊 Dashboard")

stats = get_statistics()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Patients",
    stats["total"]
)

col2.metric(
    "Average Confidence",
    f"{stats['avg_confidence']:.2f}%"
)

col3.metric(
    "Most Common Disease",
    stats["most_common"]
)
history = get_records()

# ==========================
# Patient History
# ==========================

st.markdown("---")

col1, col2 = st.columns([4, 1])

with col1:
    st.subheader("📋 Patient History")

with col2:
    if st.button("🗑️ Clear History", type="secondary"):
        clear_records()
        st.success("Patient history cleared successfully.")
        st.rerun()

history = get_records()

if history.empty:
    st.info("No patient records found.")
else:
    st.dataframe(
        history,
        width="stretch",
        hide_index=True
    )