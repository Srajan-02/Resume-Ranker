# utils/llm_integration.py - Corrected with Streamlit Caching

import streamlit as st
import openai
import requests
import json
from transformers import (
    pipeline,
    T5ForConditionalGeneration, T5Tokenizer
)
import torch
import google.generativeai as genai
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Cached Model Loading Functions ---
# These functions are decorated with @st.cache_resource to ensure they are
# run only once, and the loaded models are stored in memory for reuse.

@st.cache_resource
def load_huggingface_text_generator():
    """Loads and caches the Hugging Face text generation pipeline."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Hugging Face text-generation model (DialoGPT-medium)...")
        return pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
    except Exception as e:
        logger.error(f"Failed to load Hugging Face text generator: {e}")
        return None

@st.cache_resource
def load_huggingface_summarizer():
    """Loads and caches the Hugging Face summarization pipeline."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Hugging Face summarization model (BART)...")
        return pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if device == "cuda" else -1
        )
    except Exception as e:
        logger.error(f"Failed to load Hugging Face summarizer: {e}")
        return None

@st.cache_resource
def load_huggingface_t5_model():
    """Loads and caches the Hugging Face T5 model and tokenizer."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Hugging Face T5 model (flan-t5-base)...")
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-base",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load Hugging Face T5 model: {e}")
        return None, None

@st.cache_resource
def load_google_gemini_model():
    """Loads and caches the Google Gemini model."""
    try:
        if not Config.GOOGLE_API_KEY:
            logger.warning("Google API key not configured.")
            return None
        logger.info("Initializing Google Gemini model...")
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        logger.info("Google Gemini model initialized successfully.")
        return model
    except Exception as e:
        logger.error(f"Google Gemini initialization failed: {e}")
        return None

# --- Main Class with Unchanged Interface ---
# The class structure is preserved. The _init_ methods now call the
# cached functions above to get the models.

class FreeLLMIntegration:
    def __init__(self):
        self.provider = Config.FREE_LLM_PROVIDER if Config.USE_FREE_LLM else 'openai'
        self.available = False
        
        # Initialize based on provider
        if self.provider == 'openai' and Config.OPENAI_API_KEY:
            self._init_openai()
        elif self.provider == 'huggingface':
            self._init_huggingface()
        elif self.provider == 'google' and Config.GOOGLE_API_KEY:
            self._init_google()
        elif self.provider == 'ollama':
            self._init_ollama()
        else:
            self._init_rule_based()
            logger.info("Using rule-based explanations (no LLM)")
    
    def _init_openai(self):
        """Initialize OpenAI"""
        try:
            openai.api_key = Config.OPENAI_API_KEY
            self.available = True
            logger.info("OpenAI initialized successfully")
        except Exception as e:
            logger.error(f"OpenAI initialization failed: {e}")
            self._init_rule_based()
    
    def _init_huggingface(self):
        """Initialize Hugging Face models by calling the cached functions."""
        try:
            # These calls will be fast after the first run, as they retrieve from cache
            self.text_generator = load_huggingface_text_generator()
            self.summarizer = load_huggingface_summarizer()
            self.t5_tokenizer, self.t5_model = load_huggingface_t5_model()
            
            # The provider is available if at least one model loaded successfully
            if self.text_generator or self.summarizer or self.t5_model:
                self.available = True
                logger.info("Hugging Face models assigned successfully.")
            else:
                raise Exception("All Hugging Face models failed to load from cache.")
                
        except Exception as e:
            logger.error(f"Hugging Face initialization failed: {e}")
            self._init_rule_based()
    
    def _init_google(self):
        """Initialize Google Gemini by calling the cached function."""
        try:
            # This call will be fast after the first run, as it retrieves from cache
            self.google_model = load_google_gemini_model()
            if self.google_model:
                self.available = True
                logger.info("Google Gemini model assigned successfully.")
            else:
                raise Exception("Google Gemini model failed to load from cache.")
        except Exception as e:
            logger.error(f"Google Gemini initialization failed: {e}")
            self._init_rule_based()
    
    def _init_ollama(self):
        """Initialize Ollama (Local LLM)"""
        try:
            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                self.available = True
                logger.info("Ollama initialized successfully")
            else:
                raise Exception("Ollama not running")
        except Exception as e:
            logger.error(f"Ollama initialization failed: {e}")
            self._init_rule_based()
    
    def _init_rule_based(self):
        """Fallback to rule-based approach"""
        self.available = False
        self.provider = 'rule_based'
    
    def generate_resume_explanation(self, resume_data, jd_skills, rank):
        """Generate explanation using selected provider"""
        if self.provider == 'openai':
            return self._generate_openai_explanation(resume_data, jd_skills, rank)
        elif self.provider == 'huggingface':
            return self._generate_hf_explanation(resume_data, jd_skills, rank)
        elif self.provider == 'google':
            return self._generate_google_explanation(resume_data, jd_skills, rank)
        elif self.provider == 'ollama':
            return self._generate_ollama_explanation(resume_data, jd_skills, rank)
        else:
            return self._generate_rule_based_explanation(resume_data, jd_skills, rank)
    
    def _generate_hf_explanation(self, resume_data, jd_skills, rank):
        """Generate explanation using Hugging Face models"""
        try:
            prompt = self._create_explanation_prompt(resume_data, jd_skills, rank)
            
            # Method 1: Use T5/FLAN-T5 (Best for instruction following)
            if hasattr(self, 't5_model') and self.t5_model:
                input_text = f"Explain this resume ranking: {prompt[:500]}"
                input_ids = self.t5_tokenizer.encode(input_text, return_tensors="pt").to(self.t5_model.device)
                
                with torch.no_grad():
                    outputs = self.t5_model.generate(
                        input_ids,
                        max_length=150,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.t5_tokenizer.eos_token_id
                    )
                
                explanation = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return self._clean_hf_output(explanation)
            
            # Method 2: Use text generation pipeline
            elif hasattr(self, 'text_generator') and self.text_generator:
                prompt_text = f"Resume Analysis: This resume ranked #{rank}. "
                
                outputs = self.text_generator(
                    prompt_text,
                    max_length=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=50256
                )
                
                generated_text = outputs[0]['generated_text']
                explanation = generated_text.replace(prompt_text, "").strip()
                return self._clean_hf_output(explanation)
            
            else:
                return self._generate_rule_based_explanation(resume_data, jd_skills, rank)
                
        except Exception as e:
            logger.error(f"HF explanation generation failed: {e}")
            return self._generate_rule_based_explanation(resume_data, jd_skills, rank)
    
    def _generate_google_explanation(self, resume_data, jd_skills, rank):
        """Generate explanation using Google Gemini"""
        try:
            prompt = f"""
            Analyze this resume ranking and provide a professional explanation:
            
            Rank: #{rank}
            Score: {resume_data['final_score']:.2f}
            Matched Skills: {', '.join(resume_data['matched_skills'][:5])}
            Job Requirements: {', '.join(jd_skills[:8])}
            
            Provide a 2-3 sentence professional explanation focusing on:
            1. Why this resume received this ranking
            2. Key strengths or weaknesses
            3. Specific skill matches
            
            Keep it concise and professional.
            """
            
            response = self.google_model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Google explanation generation failed: {e}")
            return self._generate_rule_based_explanation(resume_data, jd_skills, rank)
    
    def _generate_ollama_explanation(self, resume_data, jd_skills, rank):
        """Generate explanation using Ollama"""
        try:
            prompt = self._create_explanation_prompt(resume_data, jd_skills, rank)
            
            payload = {
                "model": "llama2",  # or "mistral", "codellama", etc.
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 200
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                raise Exception(f"Ollama request failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama explanation generation failed: {e}")
            return self._generate_rule_based_explanation(resume_data, jd_skills, rank)
    
    def _generate_openai_explanation(self, resume_data, jd_skills, rank):
        """Generate explanation using OpenAI"""
        try:
            prompt = self._create_explanation_prompt(resume_data, jd_skills, rank)
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert HR assistant who explains resume rankings clearly and professionally."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI explanation generation failed: {e}")
            return self._generate_rule_based_explanation(resume_data, jd_skills, rank)
    
    def _create_explanation_prompt(self, resume_data, jd_skills, rank):
        """Create prompt for explanation generation"""
        return f"""
        Explain why this resume ranked #{rank} for the given job requirements.
        
        Resume Performance:
        - Overall Score: {resume_data['final_score']:.2f}
        - Matched Skills: {', '.join(resume_data['matched_skills'][:5])}
        - Resume Skills: {', '.join(resume_data['resume_skills'][:5])}
        
        Job Requirements: {', '.join(jd_skills[:10])}
        
        Provide a clear, professional explanation in 2-3 sentences focusing on:
        1. Key strengths that led to this ranking
        2. Specific skill matches
        3. Areas for improvement (if applicable)
        
        Keep it concise and actionable.
        """
    
    def _clean_hf_output(self, text):
        """Clean Hugging Face model output"""
        text = text.replace("</s>", "").replace("<pad>", "").strip()
        if not text:
            return "Analysis completed based on skill matching and content relevance."
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        if text and text[-1] not in '.!?':
            text += '.'
        return text
    
    def _generate_rule_based_explanation(self, resume_data, jd_skills, rank):
        """Generate explanation using rule-based approach"""
        score = resume_data['final_score']
        matched_skills = resume_data['matched_skills']
        
        if rank == 1:
            explanation = f"🏆 **Top Ranked Resume** - This resume scored {score:.2f} and ranked #1 because it "
        else:
            explanation = f"📊 **Rank #{rank}** - This resume scored {score:.2f} and "
        
        if len(matched_skills) >= 5:
            explanation += f"demonstrates excellent alignment with {len(matched_skills)} key skills including {', '.join(matched_skills[:3])}. "
        elif len(matched_skills) >= 3:
            explanation += f"shows good skill alignment with {len(matched_skills)} matched skills: {', '.join(matched_skills)}. "
        else:
            explanation += f"has limited skill overlap with only {len(matched_skills)} matched skills. "
        
        if resume_data.get('semantic_similarity', 0) > 0.7:
            explanation += "The resume content is highly relevant to the job description."
        elif resume_data.get('semantic_similarity', 0) > 0.5:
            explanation += "The resume shows moderate relevance to the job requirements."
        else:
            explanation += "Consider adding more job-relevant keywords and experiences."
        
        return explanation
    
    def generate_resume_summary(self, resume_text, job_description):
        """Generate resume summary using selected provider"""
        if self.provider == 'huggingface':
            return self._generate_hf_summary(resume_text, job_description)
        elif self.provider == 'google':
            return self._generate_google_summary(resume_text, job_description)
        elif self.provider == 'ollama':
            return self._generate_ollama_summary(resume_text, job_description)
        elif self.provider == 'openai':
            return self._generate_openai_summary(resume_text, job_description)
        else:
            return self._generate_rule_based_summary(resume_text)
    
    def _generate_hf_summary(self, resume_text, job_description):
        """Generate summary using Hugging Face models"""
        try:
            if hasattr(self, 'summarizer') and self.summarizer:
                combined_text = f"Job: {job_description[:200]} Resume: {resume_text[:800]}"
                summary_result = self.summarizer(combined_text, max_length=150, min_length=50, do_sample=False)
                summary = summary_result[0]['summary_text']
                
                return f"""
                **Resume Analysis Summary:**
                📊 **Key Findings:** {summary}
                """
            else:
                return self._generate_rule_based_summary(resume_text)
        except Exception as e:
            logger.error(f"HF summary generation failed: {e}")
            return self._generate_rule_based_summary(resume_text)
    
    def _generate_google_summary(self, resume_text, job_description):
        """Generate summary using Google Gemini"""
        try:
            prompt = f"""
            Analyze this resume against the job description and provide:
            1. Top 3 Strengths
            2. Top 2 Areas for Improvement  
            3. Overall Fit Assessment (1-10 scale)
            
            Job Description: {job_description[:500]}...
            Resume: {resume_text[:1000]}...
            
            Keep the response concise and actionable.
            """
            response = self.google_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Google summary generation failed: {e}")
            return self._generate_rule_based_summary(resume_text)
    
    def _generate_ollama_summary(self, resume_text, job_description):
        """Generate summary using Ollama"""
        try:
            prompt = f"""
            Analyze this resume and provide a brief summary with strengths and recommendations.
            Job Requirements: {job_description[:300]}
            Resume Content: {resume_text[:800]}
            """
            payload = {
                "model": "llama2", "prompt": prompt, "stream": False,
                "options": {"temperature": 0.7, "max_tokens": 300}
            }
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=45)
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                raise Exception(f"Ollama request failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama summary generation failed: {e}")
            return self._generate_rule_based_summary(resume_text)
    
    def _generate_openai_summary(self, resume_text, job_description):
        """Generate summary using OpenAI"""
        try:
            prompt = f"""
            Analyze this resume against the job description and provide:
            1. Top 3 Strengths
            2. Top 2 Areas for Improvement
            3. Overall Fit Assessment (1-10 scale)
            
            Job Description: {job_description[:500]}...
            Resume: {resume_text[:1000]}...
            
            Keep the response concise and actionable.
            """
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert resume reviewer providing constructive feedback."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300, temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI summary generation failed: {e}")
            return self._generate_rule_based_summary(resume_text)
    
    def _generate_rule_based_summary(self, resume_text):
        """Generate basic summary using rule-based approach"""
        word_count = len(resume_text.split())
        has_email = '@' in resume_text.lower()
        has_phone = any(char.isdigit() for char in resume_text)
        has_education = any(word in resume_text.lower() for word in ['university', 'college', 'degree', 'bachelor', 'master'])
        has_experience = any(word in resume_text.lower() for word in ['experience', 'worked', 'developed', 'managed'])
        
        summary = f"""
        **Resume Analysis Summary:**
        
        📊 **Document Stats:** {word_count} words
        
        ✅ **Content Check:**
        - Contact Info: {'✓' if has_email and has_phone else '⚠️'}
        - Education: {'✓' if has_education else '⚠️'}
        - Experience: {'✓' if has_experience else '⚠️'}
        
        💡 **Quick Assessment:**
        - Resume appears {'complete' if all([has_email, has_phone, has_education, has_experience]) else 'incomplete'}
        
        🎯 **Recommendations:**
        - Ensure all key skills are prominently featured.
        - Quantify achievements where possible.
        """
        return summary

# For backward compatibility
LLMIntegration = FreeLLMIntegration