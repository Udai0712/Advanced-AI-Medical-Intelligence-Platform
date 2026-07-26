from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor

import os
from datetime import datetime


def generate_pdf(
    filename,
    patient,
    result,
    explanation,
    uploaded_image_path,
    gradcam_image_path
):

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    title = styles["Title"]
    title.alignment = TA_CENTER
    title.textColor = HexColor("#0066CC")

    heading = styles["Heading2"]
    normal = styles["BodyText"]

    elements = []

    # -------------------------------------------------
    # Title
    # -------------------------------------------------

    elements.append(
        Paragraph(
            "AI MEDICAL INTELLIGENCE PLATFORM",
            title
        )
    )

    elements.append(
        Paragraph(
            "<b>Medical Analysis Report</b>",
            heading
        )
    )

    elements.append(Spacer(1, 0.3 * inch))

    # -------------------------------------------------
    # Patient Details
    # -------------------------------------------------

    elements.append(
        Paragraph("<b>Patient Information</b>", heading)
    )

    elements.append(
        Paragraph(f"Name : {patient['name']}", normal)
    )

    elements.append(
        Paragraph(f"Age : {patient['age']}", normal)
    )

    elements.append(
        Paragraph(f"Gender : {patient['gender']}", normal)
    )

    elements.append(
        Paragraph(
            f"Generated : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            normal
        )
    )

    elements.append(Spacer(1, 0.25 * inch))

    # -------------------------------------------------
    # Uploaded Image
    # -------------------------------------------------

    if os.path.exists(uploaded_image_path):

        elements.append(
            Paragraph("<b>Uploaded Skin Image</b>", heading)
        )

        elements.append(
            Image(
                uploaded_image_path,
                width=3 * inch,
                height=3 * inch
            )
        )

        elements.append(Spacer(1, 0.25 * inch))

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    elements.append(
        Paragraph("<b>Prediction Result</b>", heading)
    )

    elements.append(
        Paragraph(
            f"<b>Disease :</b> {result['prediction']}",
            normal
        )
    )

    elements.append(
        Paragraph(
            f"<b>Confidence :</b> {result['confidence']*100:.2f}%",
            normal
        )
    )

    elements.append(Spacer(1, 0.25 * inch))

    # -------------------------------------------------
    # GradCAM
    # -------------------------------------------------

    if os.path.exists(gradcam_image_path):

        elements.append(
            Paragraph("<b>Grad-CAM Visualization</b>", heading)
        )

        elements.append(
            Image(
                gradcam_image_path,
                width=3 * inch,
                height=3 * inch
            )
        )

        elements.append(Spacer(1, 0.25 * inch))

    # -------------------------------------------------
    # AI Assistant
    # -------------------------------------------------

    elements.append(
        Paragraph("<b>AI Medical Assistant</b>", heading)
    )

    explanation = explanation.replace("\n", "<br/>")

    elements.append(
        Paragraph(explanation, normal)
    )

    elements.append(Spacer(1, 0.3 * inch))

    # -------------------------------------------------
    # Disclaimer
    # -------------------------------------------------

    elements.append(
        Paragraph("<b>Disclaimer</b>", heading)
    )

    elements.append(
        Paragraph(
            "This report was generated using Artificial Intelligence. "
            "It is intended for educational and informational purposes only. "
            "It should not be considered a substitute for professional medical diagnosis, treatment, or advice. "
            "Always consult a qualified dermatologist or healthcare professional for confirmation and treatment recommendations.",
            normal
        )
    )

    doc.build(elements)