import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import Counter

class SimilarityCalculator:
    def __init__(self, nlp_processor):
        self.nlp_processor = nlp_processor
    
    def calculate_semantic_similarity(self, job_description, resumes):
        """Calculate semantic similarity using BERT embeddings"""
        # Get embeddings
        jd_embedding = self.nlp_processor.get_text_embeddings(job_description)
        resume_embeddings = self.nlp_processor.get_text_embeddings(resumes)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(jd_embedding.reshape(1, -1), resume_embeddings)
        return similarities[0]
    
    def calculate_skill_match_score(self, jd_skills, resume_skills):
        """Calculate skill matching score"""
        jd_skills_set = set([skill.lower() for skill in jd_skills])
        resume_skills_set = set([skill.lower() for skill in resume_skills])
        
        # Exact matches
        exact_matches = jd_skills_set.intersection(resume_skills_set)
        
        # Partial matches (fuzzy matching)
        partial_matches = set()
        for jd_skill in jd_skills_set:
            for resume_skill in resume_skills_set:
                if self._is_partial_match(jd_skill, resume_skill):
                    partial_matches.add((jd_skill, resume_skill))
        
        # Calculate scores
        exact_score = len(exact_matches) / len(jd_skills_set) if jd_skills_set else 0
        partial_score = len(partial_matches) / len(jd_skills_set) if jd_skills_set else 0
        
        # Combined score (weighted)
        combined_score = (exact_score * 0.8) + (partial_score * 0.2)
        
        return {
            'exact_matches': list(exact_matches),
            'partial_matches': list(partial_matches),
            'exact_score': exact_score,
            'partial_score': partial_score,
            'combined_score': combined_score
        }
    
    def _is_partial_match(self, skill1, skill2, threshold=0.7):
        """Check if two skills are partially matching"""
        # Simple substring matching
        if skill1 in skill2 or skill2 in skill1:
            return True
        
        # Jaccard similarity for word-based matching
        words1 = set(skill1.split())
        words2 = set(skill2.split())
        
        if len(words1.union(words2)) == 0:
            return False
        
        jaccard_sim = len(words1.intersection(words2)) / len(words1.union(words2))
        return jaccard_sim >= threshold
    
    def calculate_comprehensive_score(self, job_description, resume_text, 
                                    jd_skills, resume_skills):
        """Calculate comprehensive matching score"""
        # 1. Semantic similarity (40% weight)
        semantic_sim = self.calculate_semantic_similarity(job_description, [resume_text])[0]
        
        # 2. Skill matching (40% weight)
        skill_match = self.calculate_skill_match_score(jd_skills, resume_skills)
        
        # 3. Keyword density (20% weight)
        keyword_density = self._calculate_keyword_density(job_description, resume_text)
        
        # Weighted final score
        final_score = (
            semantic_sim * 0.4 +
            skill_match['combined_score'] * 0.4 +
            keyword_density * 0.2
        )
        
        return {
            'final_score': final_score,
            'semantic_similarity': semantic_sim,
            'skill_match': skill_match,
            'keyword_density': keyword_density
        }
    
    def _calculate_keyword_density(self, job_description, resume_text):
        """Calculate keyword density score"""
        # Extract important keywords from JD
        jd_processed = self.nlp_processor.preprocess_text_for_matching(job_description)
        resume_processed = self.nlp_processor.preprocess_text_for_matching(resume_text)
        
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform([jd_processed, resume_processed])
            feature_names = vectorizer.get_feature_names_out()
            
            jd_scores = tfidf_matrix[0].toarray()[0]
            resume_scores = tfidf_matrix[1].toarray()[0]
            
            # Calculate correlation between TF-IDF scores
            correlation = np.corrcoef(jd_scores, resume_scores)[0, 1]
            return max(0, correlation) if not np.isnan(correlation) else 0
        except:
            return 0
    
    def rank_resumes(self, job_description, resumes_data):
        """Rank resumes based on job description"""
        results = []
        
        # Extract skills from job description
        jd_analysis = self.nlp_processor.extract_skills_and_keywords(job_description)
        jd_skills = jd_analysis['skills'] + jd_analysis['keywords']
        
        for resume_name, resume_text in resumes_data.items():
            # Extract skills from resume
            resume_analysis = self.nlp_processor.extract_skills_and_keywords(resume_text)
            resume_skills = resume_analysis['skills'] + resume_analysis['keywords']
            
            # Calculate comprehensive score
            score_data = self.calculate_comprehensive_score(
                job_description, resume_text, jd_skills, resume_skills
            )
            
            results.append({
                'resume_name': resume_name,
                'final_score': score_data['final_score'],
                'semantic_similarity': score_data['semantic_similarity'],
                'skill_match_score': score_data['skill_match']['combined_score'],
                'keyword_density': score_data['keyword_density'],
                'matched_skills': score_data['skill_match']['exact_matches'],
                'resume_skills': resume_skills[:10],  # Top 10 skills
                'score_breakdown': score_data
            })
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Add ranking
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return results, jd_skills