# BMSIT Admission Chatbot — RAG System (v1)

A Retrieval-Augmented Generation (RAG) based chatbot for **BMS Institute of Technology & Management** that answers admission-related queries using indexed PDF documents.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Flowchart](#system-flowchart)
- [Index Builder Flow](#index-builder-flow)
- [Router Flow](#router-flow)
- [Pipeline Flow](#pipeline-flow)
- [Answer Generation & Evaluation Flow](#answer-generation--evaluation-flow)
- [Project Structure](#project-structure)
- [Component Details](#component-details)
- [How to Run](#how-to-run)
- [Retrieval Test Questions](#retrieval-test-questions)
- [Notes](#notes)

---

## Architecture Overview

```mermaid
graph TB
    subgraph "Data Layer"
        PDF[BMSIT DATA/*.pdf]
        IDX[BMSIT INDEX/]
    end

    subgraph "Indexing"
        IB[index_builder.py]
    end

    subgraph "Query Processing"
        AG[answer_generator.py]
        RT[router.py]
        AE[ANS_EVALUATOR.py]
    end

    subgraph "Retrieval Pipelines"
        PA[PIPELINE_A<br/>Dense FAISS]
        PB[PIPELINE_B<br/>Sparse BM25]
        PC[PIPELINE_C<br/>Hybrid RRF]
        PD[PIPELINE_D<br/>CrossEncoder]
        PE[PIPELINE_E<br/>Page-Level]
    end

    subgraph "LLM"
        GROQ[Groq API<br/>gpt-oss-120b]
    end

    PDF -->|extract & chunk| IB
    IB -->|build indices| IDX
    AG -->|query| RT
    RT -->|select doc + pipeline| IDX
    RT -->|dispatch| PA
    RT -->|dispatch| PB
    RT -->|dispatch| PC
    RT -->|dispatch| PD
    RT -->|dispatch| PE
    PA -->|chunks| AG
    PB -->|chunks| AG
    PC -->|chunks| AG
    PD -->|chunks| AG
    PE -->|chunks| AG
    AG -->|generate| GROQ
    AG -->|evaluate & retry| AE
    AE -->|fallback pipelines| RT
    GROQ -->|answer| AG
```

---

## System Flowchart

```mermaid
flowchart TD
    Start([User Query]) --> Reset[Reset Session Anchor]
    Reset --> Router[run_router]

    Router --> DocSelect{Layer 1:<br/>Document Selection}
    DocSelect -->|intent match| ForcedDoc[Forced Document]
    DocSelect -->|embedding + keyword| ScoreDoc[Score All Documents]
    ForcedDoc --> AnchorCheck
    ScoreDoc --> AnchorCheck{Session<br/>Anchor Check}

    AnchorCheck -->|follow-up| KeepAnchor[Keep Anchor Doc]
    AnchorCheck -->|standalone| NewDoc[Select Best Doc]
    KeepAnchor --> PipeSelect
    NewDoc --> PipeSelect

    PipeSelect{Layer 2:<br/>Pipeline Selection}
    PipeSelect -->|structure terms| PE[Pipeline E]
    PipeSelect -->|very short query| PC[Pipeline C]
    PipeSelect -->|rare terms| PB[Pipeline B]
    PipeSelect -->|long query| PD[Pipeline D]
    PipeSelect -->|mixed signals| PC
    PipeSelect -->|focused query| PA[Pipeline A]

    PA --> Chunks{Chunks Found?}
    PB --> Chunks
    PC --> Chunks
    PD --> Chunks
    PE --> Chunks

    Chunks -->|no| NoInfo[No Information]
    Chunks -->|yes| BuildPrompt[Build RAG Prompt]
    BuildPrompt --> LLM1[LLM Generation]
    LLM1 --> Eval{Answer Complete?}
    Eval -->|yes| Rewrite[Rewrite Chatbot Tone]
    Eval -->|no| Retry[Retry Fallback Pipeline]
    Retry --> LLM2[LLM Regeneration]
    LLM2 --> Eval
    Rewrite --> Output([Final Answer])
    NoInfo --> Output
```

---

## Index Builder Flow

```mermaid
flowchart TD
    Start([Auto Index Start]) --> Scan[Scan BMSIT DATA/*.pdf]
    Scan --> Loop{More PDFs?}
    Loop -->|no| Done([All Done])
    Loop -->|yes| CheckIdx{Index Exists?}
    CheckIdx -->|yes| Loop
    CheckIdx -->|no| Extract[Extract Pages]

    Extract --> TryPyPDF[Try pypdf]
    TryPyPDF --> HasText{Text Found?}
    HasText -->|yes| AddText[Add Extracted Text]
    HasText -->|no| OCR[OCR Fallback]
    OCR --> AddText
    AddText --> ImgOCR[Image OCR on Page]
    ImgOCR --> NextPage{More Pages?}
    NextPage -->|yes| Extract
    NextPage -->|no| Chunk[Chunk Pages]

    Chunk --> SaveChunks[Save chunks.json]
    SaveChunks --> Emb[Encode Chunk Embeddings]
    Emb --> SaveEmb[Save embeddings.npy]
    SaveEmb --> FAISS[Build FAISS Index]
    FAISS --> SaveFAISS[Save faiss.index]
    SaveFAISS --> DocEmb[Build Document Embedding]
    DocEmb --> SaveDocEmb[Save doc_embedding.npy]
    SaveDocEmb --> PageIdx[Build Page Index]
    PageIdx --> SavePageIdx[Save page_index.faiss + page_metadata.json]
    SavePageIdx --> Loop
```

---

## Router Flow

```mermaid
flowchart TD
    Start([run_router]) --> Embed[Encode Query Embedding]
    Embed --> L1[Layer 1: Document Selection]

    L1 --> Intent{Intent Detection}
    Intent -->|placement| Doc1[placement data]
    Intent -->|faculty| Doc2[faculty data - Sheet1]
    Intent -->|labs| Doc3[criteria 7.docx]
    Intent -->|intake| Doc4[student_details]
    Intent -->|general| Score[Score All Docs]

    Score --> ChunkSim[Chunk Similarity 45%]
    Score --> DocSim[Document Similarity 25%]
    Score --> KWScore[Keyword Score 15%]
    Score --> NameBonus[Name Bonus 15%]
    ChunkSim --> BestDoc
    DocSim --> BestDoc
    KWScore --> BestDoc
    NameBonus --> BestDoc
    Doc1 --> BestDoc[Best Document]
    Doc2 --> BestDoc
    Doc3 --> BestDoc
    Doc4 --> BestDoc

    BestDoc --> Anchor{Anchor Active?}
    Anchor -->|yes, follow-up| Keep[Keep Anchor]
    Anchor -->|no or standalone| Switch[Switch if Better]
    Keep --> L2
    Switch --> L2

    L2[Layer 2: Pipeline Selection] --> Features[Analyze Query Features]
    Features --> Length[Query Length]
    Features --> Rarity[Term Rarity]
    Features --> Diversity[Lexical Diversity]
    Features --> Dispersion[Embedding Dispersion]
    Features --> Structure[Structure Terms]

    Length --> Decision{Pipeline Decision}
    Rarity --> Decision
    Diversity --> Decision
    Dispersion --> Decision
    Structure --> Decision

    Decision --> Memory{Check Failure Memory}
    Memory -->|avoid failed| Skip[Skip Bad Pipelines]
    Memory -->|none| Dispatch
    Skip --> Dispatch[Dispatch Pipeline]

    Dispatch --> Conf{Confidence Level}
    Conf -->|LOW| FallbackC[Try Pipeline C]
    FallbackC --> Conf2{Still LOW?}
    Conf2 -->|yes| FallbackE[Try Pipeline E]
    Conf2 -->|no| Supplement
    FallbackE --> Supplement
    Conf -->|HIGH| Supplement
    Conf -->|MEDIUM| Supplement

    Supplement --> CrossRef{Cross-Reference<br/>in Chunks?}
    CrossRef -->|yes| SuppE[Supplement with Pipeline E]
    CrossRef -->|no| Return
    SuppE --> Return[Return Results]
    FallbackC --> Return
```

---

## Pipeline Flow

```mermaid
flowchart TB
    subgraph "Pipeline A — Dense FAISS"
        A1[Query] --> A2[Encode with<br/>all-MiniLM-L6-v2]
        A2 --> A3[FAISS IndexFlatIP<br/>Search top-k]
        A3 --> A4[Return Chunks<br/>+ Confidence]
    end

    subgraph "Pipeline B — Sparse BM25"
        B1[Query] --> B2[Preprocess + Lemmatize]
        B2 --> B3[BM25Okapi Scoring]
        B3 --> B4[Rank & Filter top-k]
        B4 --> B5[Return Chunks<br/>+ Confidence]
    end

    subgraph "Pipeline C — Hybrid RRF"
        C1[Query] --> C2[Dense: FAISS Search]
        C1 --> C3[Sparse: BM25 Scoring]
        C2 --> C4[Reciprocal Rank Fusion]
        C3 --> C4
        C4 --> C5[Return Merged Chunks<br/>+ Confidence]
    end

    subgraph "Pipeline D — CrossEncoder Rerank"
        D1[Query] --> D2[FAISS Search top-30]
        D2 --> D3[CrossEncoder<br/>ms-marco-MiniLM]
        D3 --> D4[Sigmoid Normalize]
        D4 --> D5[Rank & Return top-k<br/>+ Confidence]
    end

    subgraph "Pipeline E — Page-Level"
        E1[Query] --> E2{FORCE_PAGE?}
        E2 -->|yes| E3[Return Exact Page]
        E2 -->|no| E4{LAST_RESORT?}
        E4 -->|yes| E5[Search top-3 Pages<br/>+ Cross-References]
        E4 -->|no| E6[Semantic Page Search<br/>top-k]
        E3 --> E7[Return Pages<br/>+ Confidence]
        E5 --> E7
        E6 --> E7
    end
```

---

## Answer Generation & Evaluation Flow

```mermaid
flowchart TD
    Start([User types query]) --> Reset[reset_anchor]
    Reset --> Router[run_router query]
    Router --> UseLLM{use_general_llm?}
    UseLLM -->|yes| DirectLLM[Direct LLM Call]
    UseLLM -->|no| HasChunks{Chunks Found?}
    HasChunks -->|no| Sorry[Sorry, no info]
    HasChunks -->|yes| BuildCtx[Build Context from Chunks]
    BuildCtx --> Prompt[Build RAG Prompt]
    Prompt --> Gen1[LLM Generation<br/>gpt-oss-120b]
    Gen1 --> Eval[evaluate_and_retry]

    Eval --> Complete{Answer Complete?}
    Complete -->|yes| Tone[Rewrite Chatbot Tone]
    Complete -->|no| FallbackLoop{Fallback Pipelines}

    FallbackLoop -->|try next| Retry[Retry with Fallback Pipeline]
    Retry --> Gen2[LLM Regeneration]
    Gen2 --> Complete2{Complete?}
    Complete2 -->|yes| Tone
    Complete2 -->|no| More{More Retries?}
    More -->|yes| FallbackLoop
    More -->|no| Tone

    Tone --> Final([Print Final Answer])
    DirectLLM --> Final
    Sorry --> Final
```

---

## Project Structure

```text
BMSIT_RAG -2/
|
|-- BMSIT DATA/                  # Source PDFs used for indexing
|
|-- BMSIT INDEX/                 # Auto-created per-document index folders
|   `-- <pdf_name>/
|       |-- chunks.json          # Text chunks with page metadata
|       |-- embeddings.npy       # Chunk-level embeddings
|       |-- doc_embedding.npy    # Full-document embedding
|       |-- faiss.index          # Dense chunk index
|       |-- page_index.faiss     # Page-level index
|       `-- page_metadata.json   # Full page text metadata
|
|-- INDEX_BUILDER/
|   `-- index_builder.py         # Builds all chunk/page indices from PDFs
|
|-- PIPELINES/
|   |-- PIPELINE_A.py            # Dense FAISS retrieval
|   |-- PIPELINE_B.py            # Sparse BM25 retrieval
|   |-- PIPELINE_C.py            # Hybrid FAISS + BM25 via RRF
|   |-- PIPELINE_D.py            # Dense retrieval + CrossEncoder reranking
|   |-- PIPELINE_E.py            # Page-level retrieval and cross-reference support
|   `-- pipeline_utils.py        # Shared dynamic top-k logic
|
|-- ROUTER/
|   `-- router.py                # Document routing + pipeline routing
|
|-- LLMs/
|   |-- answer_generator.py      # Main chatbot entry point
|   `-- ANS_EVALUATOR.py         # Answer completeness retry + tone rewrite
|
|-- requirements.txt
|-- README.md
`-- router_failure_log.json      # Auto-generated router failure memory
```

---

## Component Details

### Index Builder (`INDEX_BUILDER/index_builder.py`)

Scans all PDFs inside `BMSIT DATA/` and builds document-level and page-level indexes inside `BMSIT INDEX/`.

- Prefers `pypdf` text extraction (preserves table ordering)
- Falls back to `pdfplumber` if needed
- OCR fallback for scanned pages
- Image OCR on embedded page images
- Creates chunk embeddings, document embeddings, FAISS chunk indexes, and page-level indexes

### Router (`ROUTER/router.py`)

Two-layer retrieval router:

**Layer 1 — Document Selection:**
- Chunk similarity (45%)
- Document-level similarity (25%)
- Keyword overlap (15%)
- Document-name bonus for intent-aligned PDFs (15%)
- Session anchor for follow-up queries (TTL: 8 queries)

**Layer 2 — Pipeline Selection:**
- Analyzes query length, term rarity, lexical diversity, embedding dispersion
- Detects structure terms and table/image references
- Router memory avoids pipelines that failed on similar past queries
- Low-confidence fallback: Pipeline C → Pipeline E

### Pipelines

| Pipeline | Method | Best For |
|----------|--------|----------|
| **A** | Dense FAISS (cosine similarity) | Focused, single-concept queries |
| **B** | Sparse BM25 with lemmatization | Exact terms, acronyms, structured wording |
| **C** | Hybrid FAISS + BM25 via RRF | Mixed semantic + keyword queries |
| **D** | Dense FAISS + CrossEncoder rerank | Long, complex, multi-aspect questions |
| **E** | Page-level retrieval | "See page N" queries, cross-references, last resort |

### Answer Generator (`LLMs/answer_generator.py`)

Main terminal chatbot loop:
- Resets anchor for each new standalone query
- Routes query through the router
- Builds strict context-only RAG prompt
- Tells model to read tables carefully
- Prefers `CAY` / `2025-26` values for current-year questions
- Explicitly lists companies, faculty, labs when present
- Evaluates and retries if answer appears incomplete

### Answer Evaluator (`LLMs/ANS_EVALUATOR.py`)

- Checks if answer is complete (detects "not available", "I don't know" patterns)
- Retries with fallback pipelines (up to 3 retries)
- Rewrites final answer into warm admissions-chatbot tone

---

## How to Run

### Step 1: Activate the environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\rag_env\Scripts\Activate.ps1"
```

### Step 2: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Configure API key

Copy the `.env` file and add your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key_here
```

> **Important:** The `.env` file is listed in `.gitignore` and will never be committed to GitHub. Never share or commit your API key.

### Step 4: Add or update PDFs

Place all source PDFs inside:

```text
BMSIT DATA/
```

### Step 5: Build or rebuild the index

```powershell
& ".\rag_env\Scripts\python.exe" ".\INDEX_BUILDER\index_builder.py"
```

Run this whenever:
- New PDFs are added
- Existing PDFs are replaced
- Extraction or indexing logic changes

### Step 6: Start the chatbot

```powershell
& ".\rag_env\Scripts\python.exe" ".\LLMs\answer_generator.py"
```

Type `exit` to quit.

---

## Retrieval Test Questions

Use these to quickly verify retrieval:

1. What is the sanctioned intake for the program in `CAY 2025-26`?
2. Which companies have recently recruited students from this department?
3. What placement records are available for the `2025 batch`?
4. Who are the faculty members in the AI&ML department, and what are their specializations?
5. What laboratories and practical facilities are available for students in this department?

---

## Notes

- Tesseract OCR must be installed for scanned PDF support. Update the path in `INDEX_BUILDER/index_builder.py` if needed.
- The index builder supports both `BMSIT DATA` / `BMSIT_DATA` and `BMSIT INDEX` / `BMSIT_INDEX`.
- `router_failure_log.json` is auto-generated and stores failed pipeline memory for similar queries.
- If retrieval behavior changes after editing PDFs or routing logic, rebuild the indexes before testing again.

---

## Roadmap

### v1 (Current) — Core RAG Architecture
- [x] PDF extraction with OCR fallback
- [x] FAISS dense + BM25 sparse indexing
- [x] Two-layer router (document + pipeline selection)
- [x] 5 retrieval pipelines (A–E)
- [x] Answer evaluation + retry with fallback pipelines
- [x] Chatbot tone rewriting
- [x] CLI interface

### v2 (Planned) — FastAPI Backend + Web Frontend
- [ ] FastAPI REST API endpoints for chat
- [ ] Session management and conversation history
- [ ] Web-based chat UI (React/Streamlit)
- [ ] Admin dashboard for document management
- [ ] Async pipeline execution
- [ ] Rate limiting and authentication
- [ ] Docker deployment support

