import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from config import Config

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class NLPProcessor:
    def __init__(self):
        # Load models
        try:
            self.nlp = spacy.load(Config.SPACY_MODEL)
        except OSError:
            print(f"SpaCy model {Config.SPACY_MODEL} not found. Please install it:")
            print(f"python -m spacy download {Config.SPACY_MODEL}")
            self.nlp = None
        
        self.sentence_model = SentenceTransformer(Config.BERT_MODEL)
        self.stop_words = set(stopwords.words('english'))
        
        # Technical skills patterns
        self.tech_skills_patterns = {
            'programming': r'\b(python|java|javascript|c\+\+|c#|php|ruby|go|rust|kotlin|swift)\b',
            'databases': r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch|cassandra)\b',
            'frameworks': r'\b(react|angular|vue|django|flask|spring|express|laravel)\b',
            'cloud': r'\b(aws|azure|gcp|docker|kubernetes|jenkins|terraform)\b',
            'data_science': r'\b(pandas|numpy|scikit-learn|tensorflow|pytorch|matplotlib|seaborn)\b',
            'tools': r'\b(git|jira|confluence|slack|figma|adobe|photoshop)\b'
        }
    
    def extract_skills_and_keywords(self, text):
        """Extract skills and keywords from text using multiple methods"""
        skills = set()
        keywords = set()
        
        # Method 1: Pattern-based extraction
        pattern_skills = self._extract_skills_by_patterns(text)
        skills.update(pattern_skills)
        
        # Method 2: SpaCy NER and POS tagging
        if self.nlp:
            spacy_skills = self._extract_skills_with_spacy(text)
            skills.update(spacy_skills)
        
        # Method 3: TF-IDF based keyword extraction
        tfidf_keywords = self._extract_keywords_tfidf(text)
        keywords.update(tfidf_keywords)
        
        # Method 4: Domain-specific skill extraction
        domain_skills = self._extract_domain_skills(text)
        skills.update(domain_skills)
        
        return {
            'skills': list(skills),
            'keywords': list(keywords),
            'all_terms': list(skills.union(keywords))
        }
    
    def _extract_skills_by_patterns(self, text):
        """Extract skills using regex patterns"""
        skills = set()
        text_lower = text.lower()
        
        for category, pattern in self.tech_skills_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            skills.update(matches)
        
        return skills
    
    def _extract_skills_with_spacy(self, text):
        """Extract skills using SpaCy NLP"""
        skills = set()
        doc = self.nlp(text)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'LANGUAGE']:
                skills.add(ent.text.lower())
        
        # Extract noun phrases that might be skills
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Keep short phrases
                clean_chunk = re.sub(r'[^\w\s]', '', chunk.text).strip()
                if clean_chunk and len(clean_chunk) > 2:
                    skills.add(clean_chunk.lower())
        
        return skills
    
    def _extract_keywords_tfidf(self, text, max_features=50):
        """Extract keywords using TF-IDF"""
        # Preprocess text
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and 
                word not in self.stop_words and len(word) > 2]
        processed_text = ' '.join(words)
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            top_indices = tfidf_scores.argsort()[-20:][::-1]
            keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            
            return keywords
        except:
            return []
    
    def _extract_domain_skills(self, text):
        """Extract domain-specific skills"""
        skills = set()
        text_lower = text.lower()
        
        # Common technical skills
        common_skills = [
            'machine learning', 'deep learning', 'artificial intelligence',
            'data analysis', 'data visualization', 'statistical analysis',
            'project management', 'agile', 'scrum', 'devops',
            'web development', 'mobile development', 'api development',
            'version control', 'testing', 'debugging', 'optimization'
        ]
        
        for skill in common_skills:
            if skill in text_lower:
                skills.add(skill)
        
        return skills
    
    def get_text_embeddings(self, texts):
        """Generate BERT embeddings for text"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.sentence_model.encode(texts)
        return embeddings
    
    def preprocess_text_for_matching(self, text):
        """Preprocess text for better matching"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\-]', ' ', text)
        
        # Remove stop words for better matching
        words = word_tokenize(text)
        words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(words)