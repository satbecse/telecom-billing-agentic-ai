import pdfplumber
from app.utils.logging import get_logger

logger = get_logger("pdf_loader")

def extract_text_from_pdf(filepath: str) -> str:
    """
    Extracts text from a single PDF using pdfplumber.
    
    Args:
        filepath: Path to the PDF file
        
    Returns:
        Extracted text as a single string
    """
    text = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted)
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Failed to read PDF {filepath}: {e}")
        return ""
