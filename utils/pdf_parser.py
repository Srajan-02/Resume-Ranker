import fitz  # PyMuPDF
import pdfminer
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import io
import re

class PDFParser:
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt']
    
    def extract_text_from_pdf(self, pdf_file, method='pymupdf'):
        """Extract text from PDF using different methods"""
        try:
            if method == 'pymupdf':
                return self._extract_with_pymupdf(pdf_file)
            elif method == 'pdfminer':
                return self._extract_with_pdfminer(pdf_file)
            else:
                # Try both methods
                try:
                    return self._extract_with_pymupdf(pdf_file)
                except:
                    return self._extract_with_pdfminer(pdf_file)
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def _extract_with_pymupdf(self, pdf_file):
        """Extract text using PyMuPDF"""
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return self._clean_text(text)
    
    def _extract_with_pdfminer(self, pdf_file):
        """Extract text using pdfminer"""
        laparams = LAParams(
            boxes_flow=0.5,
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5
        )
        
        text = extract_text(
            io.BytesIO(pdf_file.read()),
            laparams=laparams
        )
        return self._clean_text(text)
    
    def _clean_text(self, text):
        """Clean extracted text"""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\@]', '', text)
        # Remove multiple dots
        text = re.sub(r'\.{2,}', '.', text)
        return text.strip()
    
    def extract_contact_info(self, text):
        """Extract contact information from resume text"""
        contact_info = {}
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contact_info['emails'] = emails
        
        # Phone number extraction
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        contact_info['phones'] = [phone[0] + phone[1] if phone[0] else phone[1] 
                                 for phone in phones]
        
        # LinkedIn profile
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, text.lower())
        contact_info['linkedin'] = linkedin
        
        return contact_info