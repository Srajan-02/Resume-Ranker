import sqlite3
import pandas as pd
from datetime import datetime
import json
from config import Config

class ResumeDatabase:
    def __init__(self):
        self.db_path = Config.DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Resume history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resume_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_title TEXT,
                resume_name TEXT,
                similarity_score REAL,
                matched_skills TEXT,
                ranking INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Job descriptions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                extracted_skills TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_ranking_result(self, job_title, resume_name, similarity_score, 
                           matched_skills, ranking):
        """Save ranking result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO resume_history 
            (job_title, resume_name, similarity_score, matched_skills, ranking)
            VALUES (?, ?, ?, ?, ?)
        ''', (job_title, resume_name, similarity_score, 
              json.dumps(matched_skills), ranking))
        
        conn.commit()
        conn.close()
    
    def get_resume_history(self, limit=100):
        """Get resume ranking history"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM resume_history 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', conn, params=(limit,))
        conn.close()
        return df
    
    def get_performance_analytics(self):
        """Get performance analytics for dashboard"""
        conn = sqlite3.connect(self.db_path)
        
        # Average scores by job title
        avg_scores = pd.read_sql_query('''
            SELECT job_title, AVG(similarity_score) as avg_score,
                   COUNT(*) as total_resumes
            FROM resume_history 
            GROUP BY job_title
            ORDER BY avg_score DESC
        ''', conn)
        
        conn.close()
        return avg_scores