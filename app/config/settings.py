from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    # GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    ALLOWED_MODEL_NAMES =[
        "gpt-4o-mini",
        "gpt-4o"
    ]

settings=Settings()
