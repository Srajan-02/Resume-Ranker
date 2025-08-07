from utils.nlp_processor import NLPProcessor
from utils.similarity_calculator import SimilarityCalculator
from utils.llm_integration import LLMIntegration
from database import ResumeDatabase
import pandas as pd

class ResumeRanker:
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.similarity_calculator = SimilarityCalculator(self.nlp_processor)
        self.llm_integration = LLMIntegration()
        self.database = ResumeDatabase()
    
    def rank_resumes(self, job_description, resumes_data, job_title="Unknown Position"):
        """Main function to rank resumes"""
        # Rank resumes
        results, jd_skills = self.similarity_calculator.rank_resumes(
            job_description, resumes_data
        )
        
        # Generate explanations
        for result in results:
            explanation = self.llm_integration.generate_resume_explanation(
                result, jd_skills, result['rank']
            )
            result['explanation'] = explanation
            
            # Save to database
            self.database.save_ranking_result(
                job_title=job_title,
                resume_name=result['resume_name'],
                similarity_score=result['final_score'],
                matched_skills=result['matched_skills'],
                ranking=result['rank']
            )
        
        return {
            'results': results,
            'job_skills': jd_skills,
            'summary': self._generate_ranking_summary(results)
        }
    
    def _generate_ranking_summary(self, results):
        """Generate summary of ranking results"""
        if not results:
            return "No resumes to analyze."
        
        total_resumes = len(results)
        avg_score = sum([r['final_score'] for r in results]) / total_resumes
        top_score = results[0]['final_score']
        
        summary = f"""
        **Ranking Summary:**
        - Total Resumes Analyzed: {total_resumes}
        - Average Score: {avg_score:.2f}
        - Top Score: {top_score:.2f}
        - Score Range: {results[-1]['final_score']:.2f} - {top_score:.2f}
        """
        
        return summary
    
    def get_performance_analytics(self):
        """Get analytics for dashboard"""
        return self.database.get_performance_analytics()
    
    def get_resume_history(self, limit=50):
        """Get recent resume ranking history"""
        return self.database.get_resume_history(limit)