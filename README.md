# Advanced AI Medical Intelligence Platform

## Overview
An AI-powered web application for skin disease classification using Deep Learning, Explainable AI (Grad-CAM), Large Language Models, and PDF reporting.

## Features
- Skin disease prediction
- Confidence score
- Grad-CAM visualization
- AI medical explanation (Groq LLM)
- PDF report generation
- Patient history using SQLite
- Streamlit interface
- FastAPI backend

## Technologies
- Python
- TensorFlow
- Streamlit
- FastAPI
- SQLite
- OpenCV
- Groq API
- ReportLab

## Dataset
HAM10000 Skin Lesion Dataset

## Installation

```bash
git clone https://github.com/Udai0712/Advanced-AI-Medical-Intelligence-Platform.git
cd Advanced-AI-Medical-Intelligence-Platform
pip install -r requirements.txt
streamlit run app/streamlit_app.py
