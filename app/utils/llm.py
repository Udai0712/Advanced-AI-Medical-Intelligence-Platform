import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def explain_prediction(disease, confidence):
    prompt = f"""
You are an experienced dermatologist.

Prediction:
{disease}

Confidence:
{confidence:.2f}%

Provide:

1. Disease Overview
2. Common Symptoms
3. Possible Causes
4. Treatment Options
5. Prevention Tips
6. When to Consult a Dermatologist

Keep the explanation simple and easy for patients to understand.

End with the disclaimer:
"This AI-generated explanation is for educational purposes only and should not replace professional medical advice."
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful dermatologist."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=800
    )

    return response.choices[0].message.content