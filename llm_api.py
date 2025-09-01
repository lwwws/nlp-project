import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

def call_llm_api(prompt: str, model_name: str = "gemini-2.5-flash") -> str | None:
    """
    Calls the specified Google GenAI model and returns the text response.
    Returns None if an error occurs.
    """
    print(f"--- Calling Google GenAI API ({model_name}) ---")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
            ),
        )
        return response.text
    except Exception as e:
        print(f"API Call Failed: {e}")
        return None