"""
RAG Ingestion Pipeline — Medical Report Simplifier
---------------------------------------------------
Parses PDFs using OpenDataLoader (table-aware, 100% local, no GPU),
chunks with Markdown-aware splitter, embeds via Gemini, stores in FAISS.

Install:
    # Also requires Java: sudo apt install default-jre  (or brew install java on mac)

Folder structure:
    data/
        medical/        ← past patient reports, lab reports
        insurance/      ← policy documents, coverage schedules
    .env                ← GOOGLE_API_KEY=your_key
"""
import os
import json
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR         = Path("data")
MEDICAL_DIR      = DATA_DIR / "medical"
INSURANCE_DIR    = DATA_DIR / "insurance"
INDEX_DIR        = Path("faiss_index")
FAISS_INDEX_NAME = "medical_rag"
EMBEDDING_MODEL  = "gemini-embedding-001"

# Chunk sizes tuned for medical/insurance content:
# - 512 tokens keeps a full lab table + surrounding context together
# - 100 overlap ensures values at chunk boundaries aren't split from their labels
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 100

SourceType = Literal["medical", "insurance"]


# ── Step 1: Load PDFs ──────────────────────────────────────────────────────────

def load_pdfs(directory: Path, source_type: SourceType) -> list[Document]:
    """
    Load all PDFs in a directory using OpenDataLoader.

    format="markdown" is the right choice here:
      - Lab result tables → | Test | Value | Ref | Flag | markdown tables
      - Insurance coverage tables → clean markdown tables
      - Headings, lists, reading order all preserved
      - Built-in prompt injection filtering (no hidden text attacks)
    """
    pdf_files = list(directory.glob("*.pdf"))

    if not pdf_files:
        print(f"  [!] No PDFs found in {directory}")
        return []

    print(f"  Found {len(pdf_files)} PDF(s) in {directory.name}/")

    loader = OpenDataLoaderPDFLoader(
        file_path=[str(p) for p in pdf_files],
        format="markdown",  # tables preserved as markdown — critical for lab reports
        quiet=True,
    )
    raw_docs = loader.load()

    # OpenDataLoader gives us: {'source': 'path/to/file.pdf', 'format': 'markdown', 'page': N}
    # We enrich that metadata with source_type for namespace-filtered retrieval later
    for doc in raw_docs:
        doc.metadata["source_type"] = source_type
        doc.metadata["file_name"] = Path(doc.metadata["source"]).name
        # Detect if this page likely contains a table (helps with retrieval debugging)
        doc.metadata["has_table"] = "|" in doc.page_content

    print(f"  Loaded {len(raw_docs)} pages from {source_type} PDFs")
    return raw_docs


# ── Step 2: Chunk ──────────────────────────────────────────────────────────────

def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split pages into retrieval-sized chunks, preserving Markdown structure.

    Strategy:
    - Primary: MarkdownTextSplitter — respects heading and table boundaries,
      won't split a table row in half
    - Fallback: RecursiveCharacterTextSplitter for any pages that are plain
      paragraphs with no Markdown structure

    Why this matters for medical docs:
    A lab table like:
        | Creatinine | 2.1 mg/dL | 0.7-1.3 | HIGH |
    must stay in the same chunk as its column headers — otherwise the
    retriever pulls "2.1 mg/dL" with no context about what test it refers to.
    MarkdownTextSplitter handles this correctly.
    """
    md_splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for doc in docs:
        # Use markdown splitter for pages with structure, fallback for plain text
        splitter = md_splitter if ("|" in doc.page_content or "#" in doc.page_content) else fallback_splitter
        splits = splitter.split_documents([doc])

        # Tag each chunk with a is_table_chunk flag — useful for Insurance Agent
        # which should prioritise table chunks (coverage tables, co-pay schedules)
        for chunk in splits:
            chunk.metadata["is_table_chunk"] = "|" in chunk.page_content

        chunks.extend(splits)

    return chunks


# ── Step 3: Embed + Index ──────────────────────────────────────────────────────

def build_faiss_index(chunks: list[Document]) -> FAISS:
    """
    Embed all chunks via Gemini text-embedding-004 and build FAISS index.
    Batches in groups of 100 to stay within Gemini's rate limits.
    """
    print(f"\n[3] Embedding {len(chunks)} chunks via Gemini ({EMBEDDING_MODEL}) ...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    batch_size = 100
    vectorstore = None
    total_batches = -(-len(chunks) // batch_size)  # ceiling division

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"    Batch {batch_num}/{total_batches} — {len(batch)} chunks")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    return vectorstore


# ── Retriever factory (imported by agents) ─────────────────────────────────────

def load_retriever(
    source_type: SourceType | None = None,
    table_only: bool = False,
    k: int = 6,
):
    """
    Load the FAISS index and return a filtered retriever for agent use.

    Examples
    --------
    # Medical Agent — all medical history chunks
    retriever = load_retriever(source_type="medical", k=6)

    # Insurance Agent — all policy chunks
    retriever = load_retriever(source_type="insurance", k=5)

    # Insurance Agent — coverage/co-pay table chunks only
    retriever = load_retriever(source_type="insurance", table_only=True, k=4)
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        str(INDEX_DIR / FAISS_INDEX_NAME),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    search_filter: dict = {}
    if source_type:
        search_filter["source_type"] = source_type
    if table_only:
        search_filter["is_table_chunk"] = True

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            **({"filter": search_filter} if search_filter else {}),
        },
    )


# ── Stats + Validation ─────────────────────────────────────────────────────────

def print_stats(chunks: list[Document]):
    from collections import Counter

    by_source    = Counter(c.metadata["source_type"] for c in chunks)
    by_file      = Counter(c.metadata["file_name"] for c in chunks)
    table_chunks = sum(1 for c in chunks if c.metadata.get("is_table_chunk"))

    print("\n── Ingestion stats ──────────────────────────────")
    print(f"  Total chunks    : {len(chunks)}")
    print(f"  Medical chunks  : {by_source.get('medical', 0)}")
    print(f"  Insurance chunks: {by_source.get('insurance', 0)}")
    print(f"  Table chunks    : {table_chunks} ({100*table_chunks//max(len(chunks),1)}%)")
    print(f"\n  By file:")
    for fname, count in by_file.most_common():
        print(f"      {fname}: {count} chunks")

    # Sanity check: warn if table detection is suspiciously low
    if by_source.get("medical", 0) > 0 and table_chunks < 3:
        print("\n  [!] Very few table chunks detected.")
        print("      If your reports have lab tables, verify PDFs are not scanned images.")
        print("      For scanned PDFs: pip install 'opendataloader-pdf[hybrid]' and use --force-ocr")
    print("────────────────────────────────────────────────")


def validate_retrieval(vectorstore: FAISS):
    """Quick smoke test — pull chunks for typical medical and insurance queries."""
    print("\n── Retrieval smoke test ─────────────────────────")
    test_queries = [
        ("creatinine kidney function", "medical"),
        ("hospitalization claim reimbursement", "insurance"),
    ]
    for query, source_type in test_queries:
        retriever = load_retriever(source_type=source_type, k=2)
        results = retriever.invoke(query)
        print(f"\n  Query: '{query}' [{source_type}]")
        for i, r in enumerate(results):
            preview = r.page_content[:120].replace("\n", " ")
            print(f"    [{i+1}] {r.metadata['file_name']} p{r.metadata.get('page','?')} | {preview}...")
    print("────────────────────────────────────────────────")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("  Medical Report Simplifier — RAG Ingestion")
    print("=" * 52)

    # 1. Load
    print("\n[1] Loading medical reports ...")
    medical_docs = load_pdfs(MEDICAL_DIR, source_type="medical")

    print("\n[2] Loading insurance documents ...")
    insurance_docs = load_pdfs(INSURANCE_DIR, source_type="insurance")

    all_docs = medical_docs + insurance_docs
    if not all_docs:
        print("\n[!] No documents loaded. Add PDFs to data/medical/ and data/insurance/")
        return

    # 2. Chunk
    print("\n[2b] Chunking with Markdown-aware splitter ...")
    chunks = chunk_documents(all_docs)
    print(f"     {len(all_docs)} pages → {len(chunks)} chunks")
    print_stats(chunks)

    # 3. Embed + Index
    vectorstore = build_faiss_index(chunks)

    # 4. Save
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    save_path = str(INDEX_DIR / FAISS_INDEX_NAME)
    vectorstore.save_local(save_path)
    print(f"\n[4] FAISS index saved → {save_path}/")

    # 5. Chunk manifest for debugging
    manifest_path = INDEX_DIR / "chunk_manifest.json"
    manifest = [
        {
            "preview": c.page_content[:150].replace("\n", " "),
            "is_table": c.metadata.get("is_table_chunk"),
            "source_type": c.metadata.get("source_type"),
            "file": c.metadata.get("file_name"),
            "page": c.metadata.get("page"),
        }
        for c in chunks
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[5] Chunk manifest → {manifest_path}")

    # 6. Smoke test
    validate_retrieval(vectorstore)

    print("\n✓ Ingestion complete. Ready for agents.\n")


if __name__ == "__main__":
    main()