# Updated config.py with free LLM options
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI API Key (optional, for LLM features)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Free LLM Options
    USE_FREE_LLM = True  # Set to True to use free alternatives
    FREE_LLM_PROVIDER = 'huggingface'  # Options: 'huggingface', 'ollama', 'google'
    
    # Hugging Face Configuration
    HF_API_TOKEN = os.getenv('HF_API_TOKEN', '')  
    HF_TEXT_MODEL = 'microsoft/DialoGPT-medium'  # Free text generation
    HF_SUMMARIZATION_MODEL = 'facebook/bart-large-cnn'  # For summaries
    HF_TEXT2TEXT_MODEL = 'google/flan-t5-large'  # For general text tasks
    
    # Google Gemini (Free tier available)
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    
    # Model configurations
    BERT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    SPACY_MODEL = 'en_core_web_sm'
    
    # Similarity thresholds
    MIN_SIMILARITY_SCORE = 0.3
    TOP_SKILLS_COUNT = 10
    
    # Database
    DATABASE_PATH = 'database/resume_history.db'
    
    # File upload settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx']
    
    # UI settings
    PAGE_TITLE = "AI-Powered Resume Ranker"

    PAGE_ICON = "📊"


