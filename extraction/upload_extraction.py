import os
import io
import gc
import fitz  # PyMuPDF
from PIL import Image
import torch

# Surya Predictors
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.layout import LayoutPredictor

class SuryaLayoutExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize all three predictors to None
        self.det_predictor = None
        self.rec_predictor = None
        self.layout_predictor = None

    def load_models(self):
        """Loads OCR and Layout models into VRAM."""
        print(f"Loading Surya Layout & OCR models into {self.device.upper()} VRAM...")
        
        self.det_predictor = DetectionPredictor()
        self.rec_predictor = RecognitionPredictor()
        self.layout_predictor = LayoutPredictor()
        
        print("Models active. Ready to process.")

    def unload_models(self):
        """Destroys the models and forces the GPU to release the VRAM."""
        print("Unloading models and flushing VRAM...")
        
        del self.det_predictor
        del self.rec_predictor
        del self.layout_predictor
        
        self.det_predictor = None
        self.rec_predictor = None
        self.layout_predictor = None
        
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        print("VRAM cleared.")

    # --- Context Manager ---
    def __enter__(self):
        self.load_models()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_models()

    # --- Core Logic ---
    def _process_image_with_layout(self, pil_image):
        """
        1. Detects logical layout blocks.
        2. Sorts them in reading order.
        3. Crops the image.
        4. Runs OCR on each cropped block individually.
        """
        if not self.layout_predictor or not self.rec_predictor:
            raise RuntimeError("Models are not loaded! Use a 'with' block.")

        # Step 1: Get Layout Predictions
        # This identifies Tables, Text, Headers, etc., and provides coordinates
        layout_preds = self.layout_predictor([pil_image])[0]
        boxes = layout_preds.bboxes
        
        # Step 2: Sort the Bounding Boxes
        # We sort primarily by the Y coordinate (top-to-bottom).
        # We use floor division (// 10) to create a 10-pixel "row tolerance" 
        # so blocks roughly on the same horizontal line are then sorted left-to-right (X coord).
        boxes.sort(key=lambda b: (b.bbox[1] // 10, b.bbox[0]))

        extracted_blocks = []

        # Step 3 & 4: Crop and OCR
        for idx, box_data in enumerate(boxes):
            # Extract coordinates and the type of block (e.g., 'Table', 'Text')
            x1, y1, x2, y2 = box_data.bbox
            label = box_data.label
            
            # Crop the specific section from the main image
            # We add a tiny 2-pixel pad just to ensure letters on the edge aren't sliced
            pad = 2
            crop_coords = (max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad)
            cropped_img = pil_image.crop(crop_coords)
            
            # Run OCR *only* on this cropped section
            ocr_preds = self.rec_predictor(
                [cropped_img], 
                det_predictor=self.det_predictor
            )
            
            # Extract text for this block
            block_text = "\n".join([line.text for line in ocr_preds[0].text_lines]).strip()
            
            # Only append if the OCR actually found text in the box
            if block_text:
                extracted_blocks.append(f"[{label}]\n{block_text}")

        # Join all blocks together separated by a clean double newline
        return "\n\n".join(extracted_blocks)

    def process_pdf(self, file_path):
        """Processes single-page PDFs."""
        doc = fitz.open(file_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=300)
        
        img_bytes = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        return self._process_image_with_layout(pil_image)

    def process_image(self, file_path):
        """Processes standard image files."""
        pil_image = Image.open(file_path).convert("RGB")
        return self._process_image_with_layout(pil_image)

    def extract(self, file_path):
        """Routes the file to the correct processor."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return self.process_pdf(file_path)
        elif ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            return self.process_image(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

# --- EXECUTION ---

if __name__ == "__main__":
    # Point this to your MPPKVVCL bill
    test_files = ["extraction/sample_document.jpg"] 
    
    print("--- Starting Batch Job ---")
    with SuryaLayoutExtractor() as extractor:
        for file in test_files:
            try:
                print(f"Extracting {file}...\n")
                
                text = extractor.extract(file)
                
                print("=== STRUCTURED EXTRACTED TEXT ===")
                print(text)
                print("=================================\n")
                
            except Exception as e:
                print(f"Failed on {file}: {e}")
                
    print("--- Batch Job Finished ---\n")