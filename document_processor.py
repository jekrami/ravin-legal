import logging

import fitz  # PyMuPDF
from arabic_reshaper import reshape
from bidi.algorithm import get_display
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


def correct_rtl_text(text):
    """
    Corrects visually ordered (often reversed) Persian text for logical processing.
    """
    reshaped_text = reshape(text)
    return get_display(reshaped_text)


def normalize_text(text):
    """Normalizes Persian/Arabic numerals and common character variations."""
    persian_nums = "۰۱۲۳۴۵۶۷۸۹"
    arabic_nums = "٠١٢٣٤٥٦٧٨٩"
    english_nums = "0123456789"

    translation_table = str.maketrans(
        persian_nums + arabic_nums, english_nums * 2
    )
    text = text.translate(translation_table)
    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")
    text = text.replace("\u200c", " ")
    return text


def _extract_and_normalize(pdf_path):
    """Extract raw PDF text and apply normalization + RTL correction."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    normalized_text = normalize_text(full_text)
    lines = normalized_text.split("\n")
    corrected_lines = [correct_rtl_text(line) for line in lines]
    return "\n".join(corrected_lines).strip()


def extract_full_text(pdf_path):
    """Extract and normalize full PDF text without chunking (for deep analysis)."""
    try:
        text = _extract_and_normalize(pdf_path)
        if not text:
            logger.warning("No text extracted from PDF: %s", pdf_path)
        return text
    except Exception as e:
        logger.exception("Error extracting text from %s: %s", pdf_path, e)
        return ""


def process_document(pdf_path):
    """Process an uploaded PDF: extract, normalize, and chunk text for RAG."""
    try:
        final_text = _extract_and_normalize(pdf_path)
        if not final_text:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "،", " ", ""],
        )
        return text_splitter.split_text(final_text)
    except Exception as e:
        logger.exception("Error processing document %s: %s", pdf_path, e)
        return []
