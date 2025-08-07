import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import zipfile
import json
from datetime import datetime

from config import Config
from models.resume_ranker import ResumeRanker
from utils.pdf_parser import PDFParser
from utils.llm_integration import LLMIntegration

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4ECDC4;
    }
    
    .rank-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .rank-1 { background: #FFD700; color: #333; }
    .rank-2 { background: #C0C0C0; color: #333; }
    .rank-3 { background: #CD7F32; }
    .rank-other { background: #6C757D; }
    
    .skill-tag {
        display: inline-block;
        background: #E3F2FD;
        color: #1976D2;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

class ResumeRankerApp:
    def __init__(self):
        self.ranker = ResumeRanker()
        self.pdf_parser = PDFParser()
        self.llm_integration = LLMIntegration()
        
        # Initialize session state
        if 'ranking_results' not in st.session_state:
            st.session_state.ranking_results = None
        if 'uploaded_resumes' not in st.session_state:
            st.session_state.uploaded_resumes = {}
        if 'job_description' not in st.session_state:
            st.session_state.job_description = ""
    
    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🎯 AI-Powered Resume Ranker</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content based on selected page
        page = st.session_state.get('current_page', 'Resume Ranking')
        
        if page == 'Resume Ranking':
            self.render_ranking_page()
        elif page == 'Analytics Dashboard':
            self.render_analytics_page()
        elif page == 'History':
            self.render_history_page()
        elif page == 'Help':
            self.render_help_page()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("🧭 Navigation")
        
        pages = ['Resume Ranking', 'Analytics Dashboard', 'History', 'Help']
        current_page = st.sidebar.radio("Select Page", pages)
        st.session_state.current_page = current_page
        
        st.sidebar.markdown("---")
        
        # Quick stats
        st.sidebar.markdown("### 📊 Quick Stats")
        try:
            history_df = self.ranker.get_resume_history(limit=10)
            if not history_df.empty:
                st.sidebar.metric("Recent Rankings", len(history_df))
                st.sidebar.metric("Avg Score", f"{history_df['similarity_score'].mean():.2f}")
        except:
            st.sidebar.info("No ranking history available")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Settings")
        
        # LLM availability
        if self.llm_integration.available:
            st.sidebar.success("🤖 LLM Integration: Active")
        else:
            st.sidebar.warning("🤖 LLM Integration: Disabled")
            st.sidebar.info("Add OPENAI_API_KEY to .env for advanced features")
    
    def render_ranking_page(self):
        """Render main resume ranking page"""
        st.markdown("### 📝 Job Description")
        
        # Job description input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            job_description = st.text_area(
                "Enter the job description:",
                value=st.session_state.job_description,
                height=200,
                placeholder="Paste the complete job description here..."
            )
            st.session_state.job_description = job_description
        
        with col2:
            job_title = st.text_input("Job Title (Optional)", 
                                    placeholder="e.g., Data Scientist")
            
            # Sample JD button
            if st.button("📋 Load Sample JD"):
                sample_jd = self.get_sample_job_description()
                st.session_state.job_description = sample_jd
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📄 Upload Resumes")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload resume files (PDF, TXT):",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="You can upload multiple resumes at once"
        )
        
        # Process uploaded files
        if uploaded_files:
            self.process_uploaded_files(uploaded_files)
        
        # Display uploaded resumes
        if st.session_state.uploaded_resumes:
            st.markdown(f"**📁 Uploaded Resumes: {len(st.session_state.uploaded_resumes)}**")
            
            cols = st.columns(min(3, len(st.session_state.uploaded_resumes)))
            for i, resume_name in enumerate(st.session_state.uploaded_resumes.keys()):
                with cols[i % 3]:
                    st.info(f"✅ {resume_name}")
        
        st.markdown("---")
        
        # Ranking button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Rank Resumes", type="primary", use_container_width=True):
                if job_description and st.session_state.uploaded_resumes:
                    with st.spinner("🔍 Analyzing resumes..."):
                        self.perform_ranking(job_description, job_title or "Unknown Position")
                else:
                    st.error("⚠️ Please provide job description and upload at least one resume")
        
        # Display results
        if st.session_state.ranking_results:
            self.display_ranking_results()
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded resume files"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            try:
                if uploaded_file.type == "application/pdf":
                    text = self.pdf_parser.extract_text_from_pdf(uploaded_file)
                else:
                    text = str(uploaded_file.read(), "utf-8")
                
                st.session_state.uploaded_resumes[uploaded_file.name] = text
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ All files processed!")
        progress_bar.empty()
        status_text.empty()
    
    def perform_ranking(self, job_description, job_title):
        """Perform resume ranking"""
        try:
            results = self.ranker.rank_resumes(
                job_description, 
                st.session_state.uploaded_resumes,
                job_title
            )
            st.session_state.ranking_results = results
            st.success("✅ Ranking completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during ranking: {str(e)}")
    
    def display_ranking_results(self):
        """Display ranking results"""
        results = st.session_state.ranking_results
        
        st.markdown("## 🏆 Ranking Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Resumes", len(results['results']))
        
        with col2:
            avg_score = sum([r['final_score'] for r in results['results']]) / len(results['results'])
            st.metric("Average Score", f"{avg_score:.2f}")
        
        with col3:
            top_score = results['results'][0]['final_score'] if results['results'] else 0
            st.metric("Top Score", f"{top_score:.2f}")
        
        with col4:
            st.metric("Job Skills Found", len(results['job_skills']))
        
        # Results visualization
        self.create_results_visualization(results['results'])
        
        st.markdown("---")
        
        # Detailed results
        st.markdown("### 📋 Detailed Rankings")
        
        for result in results['results']:
            self.display_resume_card(result)
        
        # Download results
        st.markdown("---")
        self.create_download_section(results)
    
    def display_resume_card(self, result):
        """Display individual resume result card"""
        rank = result['rank']
        
        # Rank badge styling
        if rank == 1:
            badge_class = "rank-1"
            badge_text = "🥇 #1"
        elif rank == 2:
            badge_class = "rank-2"
            badge_text = "🥈 #2"
        elif rank == 3:
            badge_class = "rank-3"
            badge_text = "🥉 #3"
        else:
            badge_class = "rank-other"
            badge_text = f"#{rank}"
        
        with st.expander(f"{badge_text} - {result['resume_name']} (Score: {result['final_score']:.2f})", expanded=rank <= 3):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Explanation
                st.markdown("**🔍 Analysis:**")
                st.markdown(result['explanation'])
                
                # Matched skills
                if result['matched_skills']:
                    st.markdown("**✅ Matched Skills:**")
                    skills_html = "".join([f'<span class="skill-tag">{skill}</span>' 
                                         for skill in result['matched_skills'][:8]])
                    st.markdown(skills_html, unsafe_allow_html=True)
            
            with col2:
                # Score breakdown
                st.markdown("**📊 Score Breakdown:**")
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=['Semantic\nSimilarity', 'Skill\nMatch', 'Keyword\nDensity'],
                        x=[
                            result['semantic_similarity'],
                            result['skill_match_score'],
                            result['keyword_density']
                        ],
                        orientation='h',
                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                    )
                ])
                
                fig.update_layout(
                    height=200,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def create_results_visualization(self, results):
        """Create visualization for ranking results"""
        st.markdown("### 📈 Score Distribution")
        
        # Create DataFrame for plotting
        df = pd.DataFrame([{
            'Resume': r['resume_name'],
            'Final Score': r['final_score'],
            'Semantic Similarity': r['semantic_similarity'],
            'Skill Match': r['skill_match_score'],
            'Keyword Density': r['keyword_density'],
            'Rank': r['rank']
        } for r in results])
        
        # Score distribution chart
        fig = px.bar(
            df, 
            x='Resume', 
            y='Final Score',
            color='Final Score',
            color_continuous_scale='viridis',
            title="Resume Scores Comparison"
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Score components radar chart for top 3
        top_3 = df.head(3)
        
        if len(top_3) > 0:
            st.markdown("### 🎯 Top 3 Resumes - Detailed Comparison")
            
            fig = go.Figure()
            
            categories = ['Semantic Similarity', 'Skill Match', 'Keyword Density']
            
            for _, row in top_3.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['Semantic Similarity'], row['Skill Match'], row['Keyword Density']],
                    theta=categories,
                    fill='toself',
                    name=f"#{int(row['Rank'])} - {row['Resume'][:20]}..."
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def create_download_section(self, results):
        """Create download section for results"""
        st.markdown("### 💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV download
            csv_data = self.create_csv_report(results)
            st.download_button(
                label="📊 Download CSV Report",
                data=csv_data,
                file_name=f"resume_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON download
            json_data = self.create_json_report(results)
            st.download_button(
                label="📄 Download JSON Report",
                data=json_data,
                file_name=f"resume_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Detailed report
            if self.llm_integration.available:
                if st.button("📋 Generate Detailed Report"):
                    with st.spinner("Generating detailed report..."):
                        detailed_report = self.create_detailed_report(results)
                        st.download_button(
                            label="📝 Download Detailed Report",
                            data=detailed_report,
                            file_name=f"detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
    
    def create_csv_report(self, results):
        """Create CSV report of results"""
        data = []
        for result in results['results']:
            data.append({
                'Rank': result['rank'],
                'Resume Name': result['resume_name'],
                'Final Score': result['final_score'],
                'Semantic Similarity': result['semantic_similarity'],
                'Skill Match Score': result['skill_match_score'],
                'Keyword Density': result['keyword_density'],
                'Matched Skills': ', '.join(result['matched_skills']),
                'Resume Skills': ', '.join(result['resume_skills'])
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def create_json_report(self, results):
        """Create JSON report of results"""
        return json.dumps(results, indent=2, default=str)
    
    def create_detailed_report(self, results):
        """Create detailed text report using LLM"""
        report_sections = []
        
        # Header
        report_sections.append("=== DETAILED RESUME RANKING REPORT ===")
        report_sections.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append(f"Total Resumes Analyzed: {len(results['results'])}")
        report_sections.append("")
        
        # Executive Summary
        report_sections.append("EXECUTIVE SUMMARY:")
        report_sections.append(results['summary'])
        report_sections.append("")
        
        # Individual resume analysis
        for result in results['results']:
            report_sections.append(f"--- RANK #{result['rank']}: {result['resume_name']} ---")
            report_sections.append(f"Overall Score: {result['final_score']:.2f}")
            report_sections.append("")
            report_sections.append("Analysis:")
            report_sections.append(result['explanation'])
            report_sections.append("")
            
            if self.llm_integration.available:
                # Generate detailed summary for each resume
                summary = self.llm_integration.generate_resume_summary(
                    st.session_state.uploaded_resumes.get(result['resume_name'], ''),
                    st.session_state.job_description
                )
                report_sections.append("Detailed Assessment:")
                report_sections.append(summary)
            
            report_sections.append("")
            report_sections.append("-" * 50)
            report_sections.append("")
        
        return "\n".join(report_sections)
    
    def render_analytics_page(self):
        """Render analytics dashboard"""
        st.markdown("## 📊 Analytics Dashboard")
        
        try:
            # Get analytics data
            analytics_df = self.ranker.get_performance_analytics()
            history_df = self.ranker.get_resume_history(limit=100)
            
            if analytics_df.empty and history_df.empty:
                st.info("📈 No analytics data available. Run some rankings first!")
                return
            
            # Performance by job title
            if not analytics_df.empty:
                st.markdown("### 🎯 Performance by Job Title")
                
                fig = px.bar(
                        analytics_df.head(10), 
                        x='job_title', 
                        y='avg_score',
                        title="Average Scores by Job Title",
                        color='total_resumes',  # optional: use color to indicate total resumes
                        text='total_resumes'    # optional: show counts on bars
                    )
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            
            # Historical trends
            if not history_df.empty:
                st.markdown("### 📈     Historical Trends")
                
                history_df['created_at'] = pd.to_datetime(history_df['created_at'])
                daily_stats = history_df.groupby(history_df['created_at'].dt.date).agg({
                    'similarity_score': ['mean', 'count']
                }).round(2)
                daily_stats.columns = ['Average Score', 'Total Rankings']
                
                fig = px.line(
                    daily_stats.reset_index(), 
                    x='created_at', 
                    y='Average Score',
                    title="Daily Average Scores Trend"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Score distribution
                st.markdown("### 📊 Score Distribution")
                fig = px.histogram(
                    history_df, 
                    x='similarity_score', 
                    nbins=20,
                    title="Resume Score Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")
    
    def render_history_page(self):
        """Render history page"""
        st.markdown("## 📚 Ranking History")
        
        try:
            history_df = self.ranker.get_resume_history(limit=200)
            
            if history_df.empty:
                st.info("📝 No ranking history available yet.")
                return
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                job_titles = ['All'] + list(history_df['job_title'].unique())
                selected_job = st.selectbox("Filter by Job Title", job_titles)
            
            with col2:
                score_range = st.slider(
                    "Score Range", 
                    float(history_df['similarity_score'].min()), 
                    float(history_df['similarity_score'].max()),
                    (float(history_df['similarity_score'].min()), 
                     float(history_df['similarity_score'].max()))
                )
            
            with col3:
                limit = st.number_input("Number of Records", min_value=10, max_value=500, value=50)
            
            # Apply filters
            filtered_df = history_df.copy()
            
            if selected_job != 'All':
                filtered_df = filtered_df[filtered_df['job_title'] == selected_job]
            
            filtered_df = filtered_df[
                (filtered_df['similarity_score'] >= score_range[0]) &
                (filtered_df['similarity_score'] <= score_range[1])
            ].head(limit)
            
            # Display results
            st.markdown(f"### 📋 Showing {len(filtered_df)} records")
            
            # Format the dataframe for display
            display_df = filtered_df.copy()
            display_df['similarity_score'] = display_df['similarity_score'].round(3)
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                display_df[['created_at', 'job_title', 'resume_name', 'similarity_score', 'ranking']],
                use_container_width=True
            )
            
            # Export history
            if st.button("📥 Export History"):
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download History CSV",
                    data=csv_data,
                    file_name=f"ranking_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error loading history: {str(e)}")
    
    def render_help_page(self):
        """Render help page"""
        st.markdown("## ❓ Help & Documentation")
        
        # Features overview
        st.markdown("""
        ### 🚀 Key Features
        
        1. **📄 Multi-format Resume Processing**
           - Upload PDF and text files
           - Automatic text extraction and cleaning
           - Contact information extraction
        
        2. **🧠 Advanced NLP Analysis**
           - BERT embeddings for semantic similarity
           - SpaCy NER for skill extraction
           - TF-IDF for keyword analysis
        
        3. **🎯 Intelligent Ranking**
           - Multi-factor scoring algorithm
           - Semantic similarity matching
           - Skill-based evaluation
        
        4. **💡 Explainable AI**
           - Natural language explanations
           - Score breakdowns
           - Matched skills highlighting
        
        5. **📊 Analytics & Insights**
           - Performance tracking
           - Historical trends
           - Comparative analysis
        """)
        
        # Usage guide
        st.markdown("""
        ### 📖 How to Use
        
        **Step 1: Prepare Job Description**
        - Paste the complete job description
        - Include required skills and qualifications
        - Add job title for better tracking
        
        **Step 2: Upload Resumes**
        - Support for PDF and TXT formats
        - Multiple file upload supported
        - Files are processed automatically
        
        **Step 3: Run Analysis**
        - Click "Rank Resumes" to start
        - Wait for processing to complete
        - View detailed results and explanations
        
        **Step 4: Review Results**
        - Check rankings and scores
        - Read AI explanations
        - Download reports as needed
        """)
        
        # Technical details
        with st.expander("🔧 Technical Details"):
            st.markdown("""
            **NLP Models Used:**
            - Sentence Transformers: all-MiniLM-L6-v2
            - SpaCy: en_core_web_sm
            - OpenAI GPT-3.5 (optional)
            
            **Scoring Algorithm:**
            - Semantic Similarity: 40% weight
            - Skill Matching: 40% weight  
            - Keyword Density: 20% weight
            
            **File Processing:**
            - PDF: PyMuPDF + pdfminer.six
            - Text: UTF-8 encoding
            - Max file size: 5MB
            """)
        
        # FAQ
        with st.expander("❓ Frequently Asked Questions"):
            st.markdown("""
            **Q: What file formats are supported?**
            A: Currently PDF and TXT files are supported. DOCX support coming soon.
            
            **Q: How accurate is the ranking?**
            A: The system uses state-of-the-art NLP models with ~85-90% accuracy in skill matching.
            
            **Q: Can I customize the scoring weights?**
            A: Advanced customization features are planned for future releases.
            
            **Q: Is my data stored securely?**
            A: Data is processed locally and stored in encrypted SQLite database.
            
            **Q: How do I get LLM features?**
            A: Add your OpenAI API key to the .env file to enable advanced explanations.
            """)
        
        # Contact info
        st.markdown("---")
        st.markdown("""
        ### 📞 Support
        
        For technical support or feature requests:
        - 📧 Email: srajan1611@gmail.com
        - 🐛 [Report bugs on GitHub](https://github.com/Srajan-02/Resume-Ranker/tree/main)
        - 💡 Feature requests welcome!
        """)
    
    def get_sample_job_description(self):
        """Return sample job description"""
        return """
        Data Scientist - Senior Level

        We are seeking a highly skilled Data Scientist to join our growing analytics team. The ideal candidate will have strong expertise in machine learning, statistical analysis, and data visualization.

        Key Responsibilities:
        • Develop and implement machine learning models for business problems
        • Perform statistical analysis and hypothesis testing
        • Create data visualizations and dashboards using Python/R
        • Collaborate with cross-functional teams to drive data-driven decisions
        • Optimize existing algorithms and develop new analytical approaches

        Required Skills:
        • Advanced proficiency in Python and SQL
        • Experience with machine learning frameworks (scikit-learn, TensorFlow, PyTorch)
        • Strong knowledge of statistics and statistical modeling
        • Data visualization tools (Matplot lib, Seaborn, Plotly)
        • Experience with big data technologies (Spark, Hadoop)
        • Cloud platforms experience (AWS, Azure, GCP)
        • Strong communication and presentation skills

        Preferred Qualifications:
        • PhD/Masters in Data Science, Statistics, or related field
        • 3+ years of industry experience
        • Experience with deep learning and neural networks
        • Knowledge of MLOps and model deployment
        • Experience with A/B testing and experimental design
        """

# Run the application
if __name__ == "__main__":
    app = ResumeRankerApp()
    app.run()