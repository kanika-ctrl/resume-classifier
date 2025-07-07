import PyPDF2
import docx
import os
from typing import Optional

def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """Extract text from PDF file."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return None

def extract_text_from_docx(file_path: str) -> Optional[str]:
    """Extract text from DOCX file."""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return None

def extract_text_from_txt(file_path: str) -> Optional[str]:
    """Extract text from TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return None

def parse_resume_file(file_path: str) -> Optional[str]:
    """Parse resume file based on its extension."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    file_extension = file_path.lower().split('.')[-1]
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == 'docx':
        return extract_text_from_docx(file_path)
    elif file_extension == 'txt':
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file format: {file_extension}")
        return None 