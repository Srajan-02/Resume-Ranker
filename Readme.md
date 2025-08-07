# AI-Powered Resume Ranker with NLP + LLM + Explainability

A comprehensive AI system that analyzes resumes against job descriptions using advanced NLP techniques, provides intelligent rankings with explanations, and offers detailed analytics.

## 🚀 Features

- **Multi-format Resume Processing**: PDF, TXT file support
- **Advanced NLP Analysis**: BERT embeddings, SpaCy NER, TF-IDF
- **Intelligent Ranking**: Multi-factor scoring algorithm
- **Explainable AI**: Natural language explanations for rankings
- **LLM Integration**: OpenAI GPT for detailed summaries
- **Analytics Dashboard**: Performance tracking and insights
- **History Tracking**: SQLite database for result persistence
- **Interactive UI**: Clean Streamlit interface

## 📋 Prerequisites

- Python 3.8+
- pip package manager
- OpenAI API key (optional, for LLM features)

## 🛠️ Installation

```bash
git clone <repository-url>
cd ai_resume_ranker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Set up environment variables (optional)
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# Create required directories
mkdir -p database data/sample_resumes static
```

## 🚀 Usage

```bash
streamlit run app.py
```

Visit [http://localhost:8501](http://localhost:8501) and:

1. Enter job description
2. Upload resume files (PDF/TXT)
3. Click "Rank Resumes"
4. View results and download reports

## 📁 Project Structure

```
ai_resume_ranker/
├── app.py
├── config.py
├── database.py
├── requirements.txt
├── utils/
│   ├── pdf_parser.py
│   ├── nlp_processor.py
│   ├── similarity_calculator.py
│   └── llm_integration.py
├── models/
│   └── resume_ranker.py
├── static/
│   └── style.css
├── data/
└── database/
```

## ⚙️ Configuration

Edit `config.py` to customize:
- Model selection
- Similarity thresholds
- File size limits
- UI settings

## 🧪 Testing

```bash
python -m pytest tests/
python test_components.py  # Manual testing
```

## 🔧 Troubleshooting

- SpaCy model not found: `python -m spacy download en_core_web_sm`
- CUDA/GPU issues: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
- PDF parsing errors: Ensure files are not password-protected
- LLM issues: Check your `.env` file and API key validity

## 📊 Scoring Algorithm

- Semantic Similarity (40%): BERT + cosine
- Skill Matching (40%): Exact + fuzzy
- Keyword Density (20%): TF-IDF

## 🎯 Model Performance

- Accuracy: ~85-90% in skill matching
- Speed: ~2-3s per resume
- Language: English (expandable)
- File size: Max 5MB

## 🔐 Data Privacy

- Local-only processing (unless OpenAI is enabled)
- SQLite stores only metadata and scores
- Resume content not permanently stored

## 🚀 Deployment Options

### Local
```bash
streamlit run app.py
```

### Docker
```bash
docker build -t resume-ranker .
docker run -p 8501:8501 resume-ranker
```

### Cloud
- Streamlit Cloud: Link GitHub repo
- Heroku: Use `Procfile`
- AWS/GCP: Use Docker

## 📈 Optimization Tips

- **Large Files**: Use chunked/background processing
- **Model**: Quantize + cache embeddings
- **DB**: Index + archive old records

## 🛡️ Security

- Input/file validation
- API rate limiting
- Secure API key handling

## 🔄 Future Enhancements

- DOCX support
- Multi-language
- Custom scoring weights
- REST API
- Resume improvement suggestions

## 📝 API (Python)

```python
ranker = ResumeRanker()
results = ranker.rank_resumes(jd, resumes, job_title)
analytics = ranker.get_resume_analytics()
history = ranker.get_resume_history(limit=50)
```

## 📊 DB Schema (SQLite)

```sql
CREATE TABLE resume_history (
  id INTEGER PRIMARY KEY,
  job_title TEXT,
  resume_name TEXT,
  similarity_score REAL,
  matched_skills TEXT,
  ranking INTEGER,
  created_at TIMESTAMP
);
```

```sql
CREATE TABLE job_descriptions (
  id INTEGER PRIMARY KEY,
  title TEXT,
  content TEXT,
  extracted_skills TEXT,
  created_at TIMESTAMP
);
```

## 🤝 Contributing

1. Fork repo
2. Create feature branch
3. Commit + test
4. Submit PR

## 👥 Authors

- Srajan Shrivastava – Initial work
- Contributors welcome!

## 🙏 Acknowledgments

- Hugging Face
- SpaCy
- Streamlit
- OpenAI

## 📞 Support

- Email: srajan1611@gmail.com
- GitHub Issues + Discussions

---

Built with ❤️ using Python, Streamlit, and NLP
