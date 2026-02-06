#!/usr/bin/env python3
"""Build vector store from documentation files."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

console = Console()


def parse_llms_txt(content: str, source: str):
    """Parse llms.txt format into document dictionaries."""
    lines = content.split("\n")
    sections = []
    current_section = ""
    current_content = []

    for line in lines:
        if line.startswith("# "):
            if current_content:
                sections.append(
                    {
                        "content": "\n".join(current_content).strip(),
                        "metadata": {"source": source, "section": current_section},
                    }
                )
            current_section = line[2:].strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append(
            {
                "content": "\n".join(current_content).strip(),
                "metadata": {"source": source, "section": current_section},
            }
        )

    return sections


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Chunk documents into smaller pieces."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "```", ". ", " ", ""],
    )

    chunked = []
    for doc in documents:
        content = doc["content"]
        if not content.strip():
            continue

        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            chunked.append(
                {
                    "content": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "chunk": i,
                    },
                }
            )

    return chunked


def build_vectorstore(
    data_dir,
    vectorstore_path,
    embedding_model,
    ollama_base_url,
    chunk_size,
    chunk_overlap,
    batch_size,
):
    """Build the FAISS vector store."""

    # Load documents
    console.print("[cyan]📂 Loading documentation...[/cyan]")
    all_docs = []

    data_path = Path(data_dir)
    for txt_file in sorted(data_path.glob("*_llms*.txt")):
        console.print(f"   {txt_file.name}")
        content = txt_file.read_text()
        source = txt_file.stem.replace("_llms", "").replace("_full", "")
        docs = parse_llms_txt(content, source)
        all_docs.extend(docs)

    console.print(f"\n[green]✓[/green] Loaded {len(all_docs)} sections")

    # Chunk documents
    console.print(
        f"\n[cyan]✂️  Chunking documents (size={chunk_size}, overlap={chunk_overlap})...[/cyan]"
    )
    chunks = chunk_documents(all_docs, chunk_size, chunk_overlap)
    console.print(f"[green]✓[/green] Created {len(chunks)} chunks")

    # Create embeddings and vectorstore
    console.print(f"\n[cyan]🧮 Creating embeddings with Ollama ({embedding_model})...[/cyan]")
    console.print(f"   Base URL: {ollama_base_url}")
    console.print(f"   Batch size: {batch_size}")

    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=ollama_base_url,
    )

    # Convert to LangChain documents
    from langchain_core.documents import Document
    langchain_docs = [
        Document(page_content=chunk["content"], metadata=chunk["metadata"])
        for chunk in chunks
    ]

    # Process in batches with progress bar
    vectorstore = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Processing {len(langchain_docs)} documents...",
            total=len(langchain_docs),
        )

        for i in range(0, len(langchain_docs), batch_size):
            batch = langchain_docs[i:i + batch_size]

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                batch_store = FAISS.from_documents(batch, embeddings)
                vectorstore.merge_from(batch_store)

            progress.update(task, advance=len(batch))

    # Save vectorstore
    console.print(f"\n[cyan]💾 Saving vector store to {vectorstore_path}...[/cyan]")
    vectorstore.save_local(str(vectorstore_path))
    console.print("[green]✓[/green] Vector store saved")

    console.print(
        f"\n[bold green]✅ Complete![/bold green] Built index with {len(langchain_docs)} chunks"
    )


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector store")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--vectorstore-path", default="data/vectorstore", help="Output path"
    )
    parser.add_argument(
        "--embedding-model", default="nomic-embed-text", help="Embedding model"
    )
    parser.add_argument(
        "--ollama-base-url", default="http://localhost:11434", help="Ollama URL"
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")

    args = parser.parse_args()

    try:
        build_vectorstore(
            data_dir=args.data_dir,
            vectorstore_path=args.vectorstore_path,
            embedding_model=args.embedding_model,
            ollama_base_url=args.ollama_base_url,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
