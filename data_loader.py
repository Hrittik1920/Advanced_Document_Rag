import os
import io
import re
import asyncio
import nest_asyncio
import cv2
import pytesseract
import numpy as np
import pandas as pd
from langchain_core.documents import Document
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from docx import Document as DocxDocument
from docx.oxml.ns import qn
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import uuid
import fitz

from llm_clients import query_ollama

nest_asyncio.apply()

# ---------------------------------------------------------------------------
# Heading / topic-aware splitter
# ---------------------------------------------------------------------------

class HeaderAwareTextSplitter:
    """
    Splits text into chunks that respect section boundaries (headers/topics).

    Strategy:
      1. Try to detect headers via the format-specific extractor supplied by
         each loader (returns a list of (header, body) tuples).
      2. If a resulting section body is still very large, apply a lightweight
         paragraph-level fallback splitter *with minimal overlap only on those
         sub-splits* — not globally.
      3. If no headers are found at all, fall back to paragraph splitting.

    Overlap is intentionally removed at the section level.  A small overlap
    (FALLBACK_OVERLAP) is applied only when a single section must be further
    subdivided, so that long sections don't lose context at split boundaries.
    """

    MAX_SECTION_CHARS = 1200   # Sections larger than this get sub-split
    FALLBACK_OVERLAP  = 80     # Overlap used *only* inside over-long sections

    def __init__(self):
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.MAX_SECTION_CHARS,
            chunk_overlap=self.FALLBACK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
    def _generate_chunk_id(self, text: str, meta: dict) -> str:
        """Generates a deterministic ID based on source and content."""
        source = meta.get("source", "unknown")
        # Hash the source and the first 100 chars of content
        unique_string = f"{source}::{text[:100]}"
        return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()[:12]
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_sections(
        self,
        sections: list[tuple[str, str]],   # [(header, body), ...]
        base_metadata: dict,
        is_table: bool = False,
    ) -> list[Document]:
        """
        Convert pre-parsed (header, body) pairs into Documents.
        Applies sub-splitting only when a section body exceeds MAX_SECTION_CHARS.
        """
        docs = []
        for header, body in sections:
            body = body.strip()
            if not body:
                continue

            meta = {**base_metadata, "section": header} if header else base_metadata.copy()
            meta["is_table"] = is_table
            contextualized_body = f"[{header}]\n{body}" if header else body
            if is_table or len(body) <= self.MAX_SECTION_CHARS:
                meta["chunk_id"] = self._generate_chunk_id(body, meta)
                docs.append(Document(page_content=contextualized_body, metadata=meta))
            else:
                # Sub-split oversized sections; keep the header in metadata
                sub_docs = self._fallback_splitter.create_documents(
                    [body], metadatas=[meta]
                )
                for idx, sub_doc in enumerate(sub_docs):
                    sub_chunk_text = f"[{header} (Continued)]\n{sub_doc.page_content}" if header else sub_doc.page_content
                    sub_doc.page_content = sub_chunk_text
                    sub_doc.metadata["chunk_id"] = f"{self._generate_chunk_id(sub_chunk_text, meta)}_{idx}"
                docs.extend(sub_docs)

        return docs

    def split_plain_text(self, text: str, base_metadata: dict) -> list[Document]:
        """
        Detect headers in raw text, then call split_sections.
        Falls back to paragraph grouping if no headers are found.
        """
        sections = _detect_text_headers(text)
        if sections:
            return self.split_sections(sections, base_metadata)

        # No headers — split by paragraphs, merge small ones
        sections = _paragraph_sections(text)
        return self.split_sections(sections, base_metadata)


# ---------------------------------------------------------------------------
# Header detection helpers (module-level, no state needed)
# ---------------------------------------------------------------------------

# Matches: "# Header", "## Sub", "### Sub-sub"
_MARKDOWN_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# Matches: "1.2.3  Some Title", "CHAPTER 3 — INTRO", all-caps short lines
_GENERIC_HEADER_RE = re.compile(
    r"^(?:"
    # Numbered: "1.2 Section Title" (Allows for extra spaces from OCR)
    r"(?:\d+[.\d]+\s+[A-Z].{3,60})" 
    # ALL CAPS line (Allows numbers and symbols like '&' often found in headers)
    r"|(?:[A-Z0-9][A-Z0-9\s\-&]{4,60}[A-Z0-9])" 
    # Keyword + number (Accounts for OCR missing the space, e.g., "Chapter3")
    r"|(?:(?:Chapter|Section|Part|Appendix)\s*[\dIVXivx]+.?)" 
    r")$",
    re.MULTILINE | re.IGNORECASE, # Added ignorecase for 'chapter' vs 'Chapter'
)


def _detect_text_headers(text: str) -> list[tuple[str, str]]:
    """
    Returns [(header_label, section_body), ...] or [] if no headers found.
    Tries Markdown headers first, then generic pattern headers.
    """
    # --- Markdown-style ---
    matches = list(_MARKDOWN_HEADER_RE.finditer(text))
    if len(matches) >= 2:
        return _build_sections_from_matches(text, matches, label_group=2)

    # --- Generic / academic style ---
    matches = list(_GENERIC_HEADER_RE.finditer(text))
    if len(matches) >= 2:
        return _build_sections_from_matches(text, matches, label_group=0)

    return []


def _build_sections_from_matches(
    text: str,
    matches: list,
    label_group: int,
) -> list[tuple[str, str]]:
    sections = []
    for idx, match in enumerate(matches):
        header = match.group(label_group).strip()
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((header, body))
    return sections


def _paragraph_sections(text: str, min_chars: int = 100) -> list[tuple[str, str]]:
    """
    Splits text into paragraph blocks, merging tiny paragraphs
    with the next one to avoid single-sentence micro-chunks.
    """
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    merged: list[str] = []
    buffer = ""
    for para in raw_paras:
        buffer = (buffer + "\n\n" + para).strip() if buffer else para
        if len(buffer) >= min_chars:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] += "\n\n" + buffer   # attach trailing fragment to last chunk
        else:
            merged.append(buffer)

    return [("", block) for block in merged]


# ---------------------------------------------------------------------------
# DOCX-specific section extractor
# ---------------------------------------------------------------------------
def _is_toc_page(text: str) -> bool:
        """
        Detects if a block of text is primarily a Table of Contents.
        Looks for lines ending in multiple dots followed by numbers.
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return False
            
        # Regex: looks for 4+ dots, optional spaces, and digits at the end of a line
        toc_line_pattern = re.compile(r'\.{4,}\s*\d+$')
        
        toc_count = sum(1 for line in lines if toc_line_pattern.search(line))
        ratio = toc_count / len(lines)
        
        # If more than 25% of the page is TOC lines, drop it
        return ratio > 0.25
def _extract_docx_sections(docx: DocxDocument) -> list[tuple[str, str]]:
    """
    Walk paragraphs; whenever we hit a Heading style, start a new section.
    Returns [(heading_text, section_body), ...].
    Falls back to returning all text as one unnamed section if no headings exist.
    """
    text_sections: list[tuple[str, str]] = []
    table_sections: list[tuple[str, str]] = []
    current_heading = ""
    current_body_lines: list[str] = []

    for para in docx.paragraphs:
        style_name = para.style.name if para.style else ""
        text = para.text.strip()
        if not text:
            continue

        if style_name.startswith("Heading"):
            if current_body_lines:
                text_sections.append((current_heading, "\n".join(current_body_lines)))
            current_heading = text
            current_body_lines = []
        else:
            current_body_lines.append(text)

    if current_body_lines:
        text_sections.append((current_heading, "\n".join(current_body_lines)))

    for i, table in enumerate(docx.tables):
        data = []
        for row in table.rows:
            row_data = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            data.append(row_data)
            
        if len(data) > 1:
            df = pd.DataFrame(data[1:], columns=data[0])
            try:
                rows_per_chunk = 30
                for start_row in range(0, len(df), rows_per_chunk):
                    df_chunk = df.iloc[start_row : start_row + rows_per_chunk]
                    md_table = df_chunk.to_markdown(index=False)
                    
                    if md_table:
                        header = f"{current_heading} - Extracted Table {i+1} (Rows {start_row+1}-{start_row+len(df_chunk)})"
                        table_sections.append((header, md_table))
            except Exception as e:
                print(f"Skipping malformed DOCX table: {e}")

    return text_sections, table_sections


# ---------------------------------------------------------------------------
# PDF-specific header heuristic (applied after text extraction)
# ---------------------------------------------------------------------------

def _extract_pdf_sections(text: str) -> list[tuple[str, str]]:
    """
    Runs header detection on extracted PDF text.
    Identical to the generic text path but kept separate for clarity.
    """
    return _detect_text_headers(text) or _paragraph_sections(text)


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class MultiFormatDocumentLoader:
    """
    Load and process various document formats using header/topic-aware chunking.
    Chunks align with document structure (sections, headings) rather than
    arbitrary character counts.
    """

    def __init__(self):
        self.supported_extensions = {
            ".pdf", ".docx", ".csv", ".xlsx", ".xls", ".txt",
            ".png", ".jpg", ".jpeg", ".bmp", ".tiff"
        }
        self.splitter = HeaderAwareTextSplitter()

    # ------------------------------------------------------------------
    # OCR helpers (unchanged from original)
    # ------------------------------------------------------------------
    def _detect_table_grid_cv2(self, image_path: str) -> bool:
        """
        Quickly scans an image for horizontal and vertical grid lines.
        Returns True if a table structure is detected.
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False

            # Binarize the image (invert so lines are white, background is black)
            _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # Define kernels to detect horizontal and vertical lines
            # The length of the kernel dictates how long a line must be to be detected
            line_min_length = np.array(img).shape[1] // 10 
            
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_length, 1))
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_min_length))

            # Isolate horizontal lines
            horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
            # Isolate vertical lines
            vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)

            # Find intersections (where a horizontal and vertical line meet)
            intersections = cv2.bitwise_and(horizontal, vertical)

            # Count the intersections
            contours, _ = cv2.findContours(intersections, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            # If we have a decent number of intersections (e.g., 4+ corners), it's likely a table
            return len(contours) > 6

        except Exception as e:
            print(f"⚠️ Error during OpenCV table detection: {e}")
            return False
    def _looks_like_borderless_table(self, text: str) -> bool:
        """
        Analyzes OCR text for tabular spacing.
        Returns True if multiple lines have 3+ distinct columns separated by wide spaces.
        """
        if not text:
            return False
            
        lines = text.split('\n')
        tabular_line_count = 0
        
        for line in lines:
            # Look for 3 or more spaces in a row acting as a column divider
            columns = re.split(r'\s{3,}', line.strip())
            if len(columns) >= 3:
                tabular_line_count += 1
                
        # If 3 or more lines look like rows, flag it as a table
        return tabular_line_count >= 3
    async def _get_ocr_text_async(self, file_path: str, origin_info: str = "Image") -> str:
        ocr_text = ""
        confidence = 0
        has_grid = await asyncio.to_thread(self._detect_table_grid_cv2, file_path)
        if not has_grid:
            try:
                def run_tesseract():
                    img = cv2.imread(file_path)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        ocr_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
                        ocr_text = pytesseract.image_to_string(thresh).strip()
                        confidences = [int(c) for c in ocr_data["conf"] if int(c) != -1]
                        confidence = sum(confidences) / len(confidences) if confidences else 0
                        return ocr_text, confidence
                ocr_text, confidence = await asyncio.to_thread(run_tesseract)
            except Exception as e:
                print(f"⚠️  Tesseract failed for {origin_info}: {e}")

        if has_grid or self._should_use_vlm_fallback(ocr_text, confidence):
            reason = "Grid detected" if has_grid else "Low quality/Borderless table detected"
            print(f"🔄 {reason} for {origin_info}. Falling back to VLM...")
            try:
                vlm_prompt = (
                    "Extract all readable text from this image. "
                    "CRITICAL INSTRUCTION: If the image contains a table, grid, or structured data, "
                    "you MUST format it as a valid Markdown table using '|' to separate columns "
                    "and '-' to separate the header row. Preserve the exact row and column structure. "
                    "Return ONLY the markdown content, no conversational filler."
                )
                response_gen = query_ollama(
                    prompt=vlm_prompt+origin_info,
                    model="codez-ocr:latest",
                    image_path=file_path,
                    keep_alive=0,
                    stream=False
                )
                parts = []
                async for chunk in response_gen:
                    parts.append(chunk)
                ocr_text = "".join(parts).strip()
            except Exception as e:
                print(f"❌ VLM fallback failed: {e}")

        return ocr_text

    def _should_use_vlm_fallback(self, text: str, confidence: float) -> bool:
        if not text or len(text) < 10 or confidence < 60:
            return True
        if self._looks_like_borderless_table(text):
            return True
        words = text.split()
        if not words:
            return True
        return (sum(len(w) for w in words) / len(words)) < 2.5

    def _run_async_ocr(self, file_path: str, origin_info: str) -> str:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._get_ocr_text_async(file_path, origin_info))

    # ------------------------------------------------------------------
    # Format loaders
    # ------------------------------------------------------------------

    def load_image(self, file_path: str) -> list[Document]:
        """
        Images produce a single document — OCR output has no reliable
        heading structure to split on.
        """
        print(f"⏳ Processing image: {os.path.basename(file_path)}")
        text = self._run_async_ocr(file_path, os.path.basename(file_path))
        if not text:
            return []
        # Still attempt header-based splitting in case the image is a scanned doc
        metadata = {"source": os.path.basename(file_path), "file_type": "image"}
        return self.splitter.split_plain_text(text, metadata)

    def load_pdf(self, file_path: str) -> list[Document]:
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {file_path}")
            return []
        table_docs = [] 
        combined_text = ""
        metadata = {"source": os.path.basename(file_path), "file_type": "pdf"}
        try:
            with fitz.open(file_path) as f:
                for page_num, page in enumerate(f):
                    page_text = page.get_text().strip() or ""
                    if _is_toc_page(page_text):
                        print(f"⏩ Skipping TOC on page {page_num + 1}")
                        continue
                    tables = page.find_tables()
                    if tables:
                        page_text += "\n\n### Extracted Tabular Data\n\n"
                        for idx, tab in enumerate(tables):
                            try:
                                df=tab.to_pandas()
                                df.dropna(how="all", inplace=True)  # Drop empty rows
                                if not df.empty:
                                    rows_per_chunk = 30  # Safe limit for embedding models
                                    for start_row in range(0, len(df), rows_per_chunk):
                                        df_chunk = df.iloc[start_row : start_row + rows_per_chunk]
                                        md_table = df_chunk.to_markdown(index=False)
                                        
                                        if md_table:
                                            table_meta = {**metadata, "page": page_num + 1}
                                            # Label which rows are in this chunk
                                            header = f"Extracted Table {idx + 1} (Page {page_num + 1}, Rows {start_row+1}-{start_row+len(df_chunk)})"
                                            table_docs.extend(
                                                self.splitter.split_sections([(header, md_table)], table_meta, is_table=True)
                                            )
                            except Exception as e:
                                print(f"Error extracting table from PDF page: {e}")
                        combined_text += page_text + "\n\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return []

        metadata = {"source": os.path.basename(file_path), "file_type": "pdf"}

        # Digital PDF with sufficient text
        text_docs = []
        if len(combined_text.strip()) > 200:
            sections = _extract_pdf_sections(combined_text)
            text_docs = self.splitter.split_sections(sections, metadata, is_table=False)
        else:
        # Scanned PDF — OCR each page, treat each page as its own section
            print(f"⚠️  Scanned PDF detected: {os.path.basename(file_path)}")
            all_docs: list[Document] = []
            try:
                images = convert_from_path(file_path)
                for i, img in enumerate(images):
                    temp_path = f"temp_{os.getpid()}_{i}.jpg"
                    img = img.convert('RGB')
                    img.save(temp_path, "JPEG", quality=90)
                    page_text = self._run_async_ocr(temp_path, f"Page {i + 1}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    if not page_text:
                        continue
                    page_meta = {**metadata, "file_type": "pdf_scanned", "page": i + 1}
                    # Each scanned page is already a natural boundary
                    sections = _detect_text_headers(page_text) or [("", page_text)]
                    text_docs.extend(self.splitter.split_sections(sections, page_meta))
            except Exception as e:
                print(f"Error during scanned PDF OCR for {file_path}: {e}")

        return text_docs + table_docs

    def load_docx(self, file_path: str) -> list[Document]:
        try:
            docx = DocxDocument(file_path)
        except Exception as e:
            print(f"Error loading DOCX {file_path}: {e}")
            return []

        text_sections, table_sections = _extract_docx_sections(docx)
        metadata = {"source": os.path.basename(file_path), "file_type": "docx"}
        
        # Process text and tables separately to protect the tables
        docs = self.splitter.split_sections(text_sections, metadata, is_table=False)
        docs.extend(self.splitter.split_sections(table_sections, metadata, is_table=True))
        return docs

    def load_csv(self, file_path: str, rows_per_chunk: int = 250) -> list[Document]:
        """
        CSVs have no heading structure; row-group chunking is the right
        semantic boundary here — unchanged from original.
        """
        docs = []
        try:
            for i, df_chunk in enumerate(
                pd.read_csv(file_path, on_bad_lines="skip", chunksize=rows_per_chunk, low_memory=True)
            ):
                buf = io.StringIO()
                content=df_chunk.to_markdown(index=False)
                if content:
                    start = i * rows_per_chunk + 1
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": os.path.basename(file_path),
                            "rows": f"{start}-{start + len(df_chunk) - 1}",
                            "file_type": "csv_chunk",
                        },
                    ))
        except Exception as e:
            print(f"Error loading CSV {file_path}: {e}")
        return docs

    def load_excel(self, file_path: str, rows_per_chunk: int = 250) -> list[Document]:
        """
        Each sheet is treated as a named section; within a sheet,
        further splitting uses the plain-text path if the sheet is large.
        """
        docs = []
        try:
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Manually slice the dataframe into chunks
                for start_row in range(0, len(df), rows_per_chunk):
                    df_chunk = df.iloc[start_row : start_row + rows_per_chunk]
                    content = df_chunk.to_markdown(index=False)
                    
                    if content:
                        start_idx = start_row + 1
                        end_idx = start_row + len(df_chunk)
                        docs.append(Document(
                            page_content=content,
                            metadata={
                                "source": os.path.basename(file_path),
                                "sheet": sheet_name,
                                "rows": f"{start_idx}-{end_idx}",
                                "file_type": "excel_chunk",
                                "is_table": True
                            },
                        ))
        except Exception as e:
            print(f"Error loading Excel {file_path}: {e}")
        return docs

    def load_txt(self, file_path: str) -> list[Document]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error loading TXT {file_path}: {e}")
            return []

        metadata = {"source": os.path.basename(file_path), "file_type": "txt"}
        return self.splitter.split_plain_text(text, metadata)

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def load_document(self, file_path: str) -> list[Document]:
        ext = Path(file_path).suffix.lower()
        dispatch = {
            ".pdf":  self.load_pdf,
            ".docx": self.load_docx,
            ".csv":  self.load_csv,
            ".xlsx": self.load_excel,
            ".xls":  self.load_excel,
            ".txt":  self.load_txt,
            ".png":  self.load_image,
            ".jpg":  self.load_image,
            ".jpeg": self.load_image,
            ".bmp":  self.load_image,
            ".tiff": self.load_image,
        }
        loader = dispatch.get(ext)
        if loader:
            return loader(file_path)
        print(f"Unsupported file type: {ext}")
        return []
    
    
    
def dump_chunks_to_file(
    docs: list,
    output_path: str = "vector.txt",
    encoding: str = "utf-8",
    no_change: bool = False  # Default to False so it dumps by default
) -> None:
    """
    Write every document chunk along with its metadata to a plain-text file.
    """
    if no_change:
        print(f"⏩ Skipping chunk dump (no_change=True).")
        return

    # 1. Added chunk_id to KNOWN_KEYS to expose the hash you generated
    KNOWN_KEYS = ["chunk_id", "source", "file_type", "section", "page", "rows", "sheet"]
    LABELS = {
        "chunk_id":  "CHUNK ID ", # Added label
        "source":    "SOURCE   ",
        "file_type": "FILE TYPE",
        "section":   "SECTION  ",
        "page":      "PAGE     ",
        "rows":      "ROWS     ",
        "sheet":     "SHEET    ",
    }

    WIDE  = "=" * 80
    THIN  = "-" * 80

    # 2. Added errors="replace" to prevent weird OCR characters from crashing the dump
    with open(output_path, "w", encoding=encoding, errors="replace") as f:
        f.write(f"TOTAL CHUNKS: {len(docs)}\n")
        f.write(WIDE + "\n\n")

        for idx, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            content = doc.page_content or ""

            f.write(WIDE + "\n")
            f.write(f"CHUNK {idx}\n")
            f.write(THIN + "\n")

            # Known metadata fields
            for key in KNOWN_KEYS:
                if key in meta:
                    label = LABELS[key]
                    f.write(f"{label}: {meta[key]}\n")

            # Char count
            f.write(f"CHARS    : {len(content)}\n")

            # Any extra/unknown metadata keys
            extra = {k: v for k, v in meta.items() if k not in KNOWN_KEYS}
            if extra:
                f.write(f"OTHER    : {extra}\n")

            f.write(THIN + "\n")
            f.write(content)
            f.write("\n" + WIDE + "\n\n")

    print(f"✅ Dumped {len(docs)} chunks → {output_path}")