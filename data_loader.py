import os
import pandas as pd
from langchain_core.documents import Document
import PyPDF2
from docx import Document as DocxDocument
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import io
import cv2
import pytesseract

class MultiFormatDocumentLoader:
    """
    Load and process various document formats with significant optimizations
    for creating fewer, more meaningful document chunks.
    """

    def __init__(self, chunk_size=2000, chunk_overlap=200):
        self.supported_extensions = {
            ".pdf", ".docx", ".csv", ".xlsx", ".xls", ".txt",
            ".png", ".jpg", ".jpeg", ".bmp", ".tiff"  # <-- ADDED IMAGE EXTENSIONS
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def _create_documents_from_text(self, text: str, metadata: dict) -> list[Document]:
        """Splits a single large text into multiple Document objects."""
        if not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        docs = []
        for i, chunk_text in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata["chunk_number"] = i + 1
            docs.append(Document(page_content=chunk_text, metadata=doc_metadata))
        return docs

    def load_txt(self, file_path: str) -> list[Document]:
        """Loads a .txt file and processes it."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
        metadata = {"source": os.path.basename(file_path), "file_type": "txt"}
        return self._create_documents_from_text(text, metadata)

    def load_pdf(self, file_path: str) -> list[Document]:
        """
        OPTIMIZED: Extracts text from all pages of a PDF, combines it,
        and then splits the combined text into chunks.
        """
        full_text = ""
        try:
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    full_text += page.extract_text() or ""
                    full_text += "\n" # Add a separator between pages
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return []
        
        metadata = {"source": os.path.basename(file_path), "file_type": "pdf"}
        return self._create_documents_from_text(full_text, metadata)

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

    def load_image(self, file_path: str) -> list[Document]:
        """
        Extracts text from an image using OpenCV and Tesseract OCR,
        and then splits the text into document chunks.
        """
        full_text = ""
        try:
            print(f"Running Tesseract OCR for {os.path.basename(file_path)}...")
            img = cv2.imread(file_path)
            
            if img is None:
                print(f"Error loading Image {file_path}: Image is unreadable or not found.")
                return []
                
            # Convert to grayscale and apply Otsu's thresholding for better OCR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text
            full_text = pytesseract.image_to_string(thresh).strip()
            
            # NOTE: If you still want your VLM fallback logic, you can check 
            # if not full_text here and call your synchronous VLM function.

        except Exception as e:
            print(f"Error loading Image {file_path}: {e}")
            return []

        metadata = {"source": os.path.basename(file_path), "file_type": "image"}
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
        Processes each sheet in an Excel file. For simplicity, we still process
        row-by-row here, but a CSV-like chunking could be added if needed.
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
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}: # <-- ADDED THIS BLOCK
            return self.load_image(file_path)
            
        print(f"Unsupported file type: {ext}")
        return []