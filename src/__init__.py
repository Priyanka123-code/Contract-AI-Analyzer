"""Contract Analysis LLM module - Using FREE Gemini + Trained ML Model (93% accuracy)"""

import os

import google.generativeai as genai
import joblib
from dotenv import load_dotenv

load_dotenv()

# ==================== GEMINI SETUP ====================
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

# Using the model that is currently working for you
client = genai.GenerativeModel(
    model_name="gemini-3-flash-preview",
    generation_config={
        "temperature": 0.1,
        "response_mime_type": "application/json",
    },
)

print("Gemini client initialized with gemini-3-flash-preview")

# ==================== LOAD TRAINED ML MODEL ====================
try:
    ml_model = joblib.load("model_train/cuad_clause_classifier.pkl")
    ml_vectorizer = joblib.load("model_train/cuad_tfidf_vectorizer.pkl")
    print("Trained ML model (93% accuracy) loaded successfully")
except Exception as e:
    ml_model = None
    ml_vectorizer = None
    print(f"ML model not found: {e}; falling back to Gemini only")

# CRITICAL: Imports at the bottom to prevent circular import errors
from .ext import extract_text_from_pdf
from .pre import ContractMetadata, extract_contract_metadata
from .utils import CUAD_CATEGORIES, ClauseAnalysis, classify_clauses, classify_clauses_ml
from .chroma import add_contract_to_vector_db, get_collection

__all__ = [
    "client",
    "extract_text_from_pdf",
    "extract_contract_metadata",
    "ContractMetadata",
    "classify_clauses",
    "classify_clauses_ml",
    "CUAD_CATEGORIES",
    "ClauseAnalysis",
    "add_contract_to_vector_db",
    "get_collection",
    "ml_model",
    "ml_vectorizer",
]
