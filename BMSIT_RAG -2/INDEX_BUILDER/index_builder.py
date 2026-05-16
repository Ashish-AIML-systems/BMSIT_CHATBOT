"""
BMSIT AUTO DOCUMENT INDEX BUILDER
----------------------------------
Scans BMSIT DATA / BMSIT_DATA folder for files.

Assumption:
→ ALL files inside folder are PDFs (user ensured)
"""

import os
import io
import json
import sys

BUNDLED_SITE_PACKAGES = r"C:\Users\ashis\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages"
if BUNDLED_SITE_PACKAGES not in sys.path:
    sys.path.append(BUNDLED_SITE_PACKAGES)
import numpy as np
import faiss
import pdfplumber
import pytesseract
from PIL import Image
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ===============================
# TESSERACT CONFIG
# ===============================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ===============================
# PATH CONFIGURATION (FIXED)
# ===============================

# 🔥 Move ONE LEVEL UP from PIPELINES
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Support both naming styles
DOC_DIR = os.path.join(BASE_DIR, "BMSIT DATA")
if not os.path.exists(DOC_DIR):
    DOC_DIR = os.path.join(BASE_DIR, "BMSIT_DATA")

INDEX_DIR = os.path.join(BASE_DIR, "BMSIT INDEX")
if not os.path.exists(INDEX_DIR):
    INDEX_DIR = os.path.join(BASE_DIR, "BMSIT_INDEX")

os.makedirs(DOC_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


# ===============================
# MODEL
# ===============================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ===============================
# CHUNKING
# ===============================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 150

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
)

OCR_RESOLUTION = 300
MIN_OCR_CHARS = 30


# ===============================
# OCR (IMAGES)
# ===============================

def ocr_images_on_page(page):
    texts = []
    try:
        for img in page.images:
            data = img.get("stream")
            if data is None:
                continue

            try:
                image = Image.open(io.BytesIO(data)).convert("RGB")
            except:
                continue

            txt = pytesseract.image_to_string(image, config="--psm 6").strip()

            if len(txt) >= MIN_OCR_CHARS:
                texts.append(f"[IMAGE]: {txt}")

    except Exception as e:
        print(f"Image OCR error: {e}")

    return "\n".join(texts)


# ===============================
# PDF EXTRACTION
# ===============================

def extract_pages(pdf_path):
    pages = []

    reader = PdfReader(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            num = i + 1
            parts = []

            # Prefer pypdf text extraction because it preserves table ordering
            # better than pdfplumber on these accreditation-style PDFs.
            txt = ""
            try:
                txt = reader.pages[i].extract_text() or ""
            except Exception:
                txt = ""

            if not txt.strip():
                txt = page.extract_text() or ""

            if txt and txt.strip():
                parts.append(txt.strip())
                print(f"Page {num}: text extracted")
            else:
                print(f"Page {num}: OCR fallback")

                try:
                    img = page.to_image(resolution=OCR_RESOLUTION).original
                    ocr = pytesseract.image_to_string(img, config="--psm 6").strip()

                    if len(ocr) >= MIN_OCR_CHARS:
                        parts.append(f"[OCR]: {ocr}")
                except Exception as e:
                    print(f"OCR failed: {e}")

            # IMAGE OCR
            img_txt = ocr_images_on_page(page)
            if img_txt:
                parts.append(img_txt)

            full = "\n\n".join(parts).strip()
            if full:
                pages.append((num, full))

    return pages


# ===============================
# CHUNKING
# ===============================

def chunk_pages(pages, name):
    chunks = []
    cid = 0

    for page, text in pages:
        for chunk in splitter.split_text(text):
            if len(chunk.strip()) < MIN_OCR_CHARS:
                continue

            chunks.append({
                "chunk_id": cid,
                "pdf_name": name,
                "page": page,
                "text": chunk.strip()
            })
            cid += 1

    return chunks


# ===============================
# PAGE INDEX
# ===============================

def build_page_index(pages, out_dir):
    path = os.path.join(out_dir, "page_index.faiss")
    meta = os.path.join(out_dir, "page_metadata.json")

    if os.path.exists(path):
        return

    records = [{"page": p, "text": t} for p, t in pages]
    texts = [r["text"] for r in records]

    emb = model.encode(texts)
    emb = np.array(emb).astype("float32")

    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, path)

    with open(meta, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Page index built ({len(records)})")


# ===============================
# BUILD INDEX
# ===============================

def build_index(pdf_path):
    name = os.path.splitext(os.path.basename(pdf_path))[0]
    out = os.path.join(INDEX_DIR, name)

    os.makedirs(out, exist_ok=True)

    if os.path.exists(os.path.join(out, "faiss.index")):
        print(f"Skipping {name}")
        return

    print(f"\nProcessing: {name}")

    pages = extract_pages(pdf_path)
    if not pages:
        print("No content found")
        return

    chunks = chunk_pages(pages, name)
    print(f"Chunks: {len(chunks)}")

    with open(os.path.join(out, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    texts = [c["text"] for c in chunks]

    emb = model.encode(texts, normalize_embeddings=True)
    emb = np.array(emb).astype("float32")

    np.save(os.path.join(out, "embeddings.npy"), emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, os.path.join(out, "faiss.index"))

    doc_emb = model.encode([" ".join(texts)], normalize_embeddings=True)
    np.save(os.path.join(out, "doc_embedding.npy"), doc_emb)

    build_page_index(pages, out)

    print(f"✓ Done: {name}")


# ===============================
# AUTO RUN
# ===============================

def auto_index():
    files = os.listdir(DOC_DIR)

    if not files:
        print("No files found in folder")
        return

    print("FILES FOUND:", files)

    for f in files:
        path = os.path.join(DOC_DIR, f)

        if os.path.isfile(path):
            try:
                build_index(path)
            except Exception as e:
                print(f"Error processing {f}: {e}")

    print("\nALL DONE 🚀")


if __name__ == "__main__":
    auto_index()