import os
import io
import asyncio
import nest_asyncio
import cv2
import pytesseract
import numpy as np
import pandas as pd
from langchain_core.documents import Document
import PyPDF2
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import your VLM client and settings
from llm_clients import query_ollama

nest_asyncio.apply()
class MultiFormatDocumentLoader:
    """
    Load and process various document formats with significant optimizations
    for creating fewer, more meaningful document chunks.
    """

    def __init__(self, chunk_size=2000, chunk_overlap=200):
        self.supported_extensions = {
            ".pdf", ".docx", ".csv", ".xlsx", ".xls", ".txt",
            ".png", ".jpg", ".jpeg", ".bmp", ".tiff"
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def _create_documents_from_text(self, text: str, metadata: dict) -> list[Document]:
        """Helper to split extracted text into Document objects."""
        if not text or not text.strip():
            return []
        # Langchain's text splitter can process a list of texts and metadatas
        return self.text_splitter.create_documents([text], metadatas=[metadata])

    # --- New Internal OCR Logic (Hybrid Tesseract + VLM) ---

    async def _get_ocr_text_async(self, file_path, origin_info="Image"):
        """Internal async method to handle the OCR logic flow."""
        ocr_text = ""
        confidence = 0
        
        try:
            # Tesseract Step
            img = cv2.imread(file_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                ocr_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
                ocr_text = pytesseract.image_to_string(thresh).strip()
                
                confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) != -1]
                confidence = sum(confidences) / len(confidences) if confidences else 0
        except Exception as e:
            print(f"⚠️ Tesseract failed for {origin_info}: {e}")

        # Quality Check & VLM Fallback
        if self._should_use_vlm_fallback(ocr_text, confidence):
            print(f"🔄 Low quality result for {origin_info}. Falling back to VLM...")
            try:
                ocr_text = await query_ollama(
                    prompt="Extract all readable text from this image. Return only text.",
                    model="deepseek-ocr:latest",
                    image_path=file_path,
                    keep_alive=0
                )
            except Exception as e:
                print(f"❌ VLM Fallback failed: {e}")
        
        return ocr_text

    def _should_use_vlm_fallback(self, text, confidence):
        """Logic to decide if OCR quality is sufficient."""
        if not text or len(text) < 10 or confidence < 60:
            return True
        
        words = text.split()
        if not words: return True
        
        avg_word_length = sum(len(w) for w in words) / len(words)
        if avg_word_length < 2.5: return True # Fragmented text
        
        return False

    def _run_async_ocr(self, file_path, origin_info):
        """Helper to run the async OCR logic in a synchronous environment."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self._get_ocr_text_async(file_path, origin_info))

    # --- Updated Loaders ---

    def load_image(self, file_path: str) -> list[Document]:
        """Extracts text using the Hybrid OCR logic."""
        print(f"⏳ Processing Image: {os.path.basename(file_path)}")
        full_text = self._run_async_ocr(file_path, os.path.basename(file_path))
        
        metadata = {"source": os.path.basename(file_path), "file_type": "image"}
        return self._create_documents_from_text(full_text, metadata)

    def load_pdf(self, file_path: str) -> list[Document]:
        """
        OPTIMIZED: Extracts text from all pages of a PDF, combines it,
        and then splits the combined text into chunks.
        """
        all_docs = []
        is_scanned = True
        total_chars = 0
        
        try:
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        total_chars += len(page_text.strip())
                        metadata = {
                            "source": os.path.basename(file_path), 
                            "file_type": "pdf",
                            "page": i 
                        }
                        page_docs = self._create_documents_from_text(page_text, metadata)
                        all_docs.extend(page_docs)
                if total_chars> 50:
                    is_scanned = False

            # 2. Fallback to OCR if scanned
            if is_scanned:
                print(f"⚠️ PDF appears scanned: {os.path.basename(file_path)}. Converting to images...")
                images = convert_from_path(file_path)
                
                for i, img in enumerate(images):
                    # Save temporary page image
                    temp_page_path = f"temp_page_{i}.png"
                    img.save(temp_page_path, "PNG")
                    
                    page_text = self._run_async_ocr(temp_page_path, f"Page {i+1}")
                    
                    if os.path.exists(temp_page_path):
                        os.remove(temp_page_path)
                        
                    
                    metadata = {
                        "source": os.path.basename(file_path), 
                        "file_type": "pdf_scanned",
                        "page": i 
                    }
                    page_docs = self._create_documents_from_text(page_text, metadata)
                    all_docs.extend(page_docs)

        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return []
        
        return all_docs

    def load_docx(self, file_path: str) -> list[Document]:
        """
        OPTIMIZED: Extracts text from a .docx file, combines it,
        and then splits the combined text.
        """
        try:
            docx = DocxDocument(file_path)
            full_text = "\n".join([p.text for p in docx.paragraphs if p.text.strip()])
        except Exception as e:
            print(f"Error loading DOCX {file_path}: {e}")
            return []
            
        metadata = {"source": os.path.basename(file_path), "file_type": "docx"}
        return self._create_documents_from_text(full_text, metadata)

    def load_csv(self, file_path: str, rows_per_chunk: int = 250) -> list[Document]:
        """
        OPTIMIZED: Loads a large CSV by grouping rows into larger documents
        for much faster processing.
        """
        docs = []
        try:
            chunk_iter = pd.read_csv(file_path, on_bad_lines='skip', chunksize=rows_per_chunk, low_memory=True)
            for i, df_chunk in enumerate(chunk_iter):
                string_buffer = io.StringIO()
                df_chunk.to_csv(string_buffer, index=False)
                chunk_content = string_buffer.getvalue()

                if chunk_content.strip():
                    start_row = i * rows_per_chunk + 1
                    end_row = start_row + len(df_chunk) - 1
                    metadata = {
                        "source": os.path.basename(file_path),
                        "rows": f"{start_row}-{end_row}",
                        "file_type": "csv_chunk",
                    }
                    docs.append(Document(page_content=chunk_content, metadata=metadata))
        except Exception as e:
            print(f"Error loading CSV {file_path}: {e}")
        return docs

    def load_excel(self, file_path: str) -> list[Document]:
        """
        Processes each sheet in an Excel file.
        """
        docs = []
        try:
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # Combine all rows of a sheet into one text block
                sheet_text = df.to_string()
                metadata = {
                    "source": os.path.basename(file_path),
                    "sheet": sheet_name,
                    "file_type": "excel",
                }
                docs.extend(self._create_documents_from_text(sheet_text, metadata))
        except Exception as e:
            print(f"Error loading Excel {file_path}: {e}")
        return docs

    # --- TXT Loader (ADDED) ---
    def load_txt(self, file_path: str) -> list[Document]:
        """Extracts text from a standard .txt file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        except Exception as e:
            print(f"Error loading TXT {file_path}: {e}")
            return []
        
        metadata = {"source": os.path.basename(file_path), "file_type": "txt"}
        return self._create_documents_from_text(full_text, metadata)

    def load_document(self, file_path: str) -> list[Document]:
        """Public method to load a single document based on its extension."""
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self.load_pdf(file_path)
        if ext == ".docx":
            return self.load_docx(file_path)
        if ext == ".csv":
            return self.load_csv(file_path)
        if ext in {".xlsx", ".xls"}:
            return self.load_excel(file_path)
        if ext == ".txt":
            return self.load_txt(file_path)
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}: 
            return self.load_image(file_path)
            
        print(f"Unsupported file type: {ext}")
        return []