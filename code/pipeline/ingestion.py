import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from config import DATA_DIR, logger

class Document:
    def __init__(self, filepath: str, content: str, metadata: dict = None):
        self.filepath = filepath
        self.content = content
        self.metadata = metadata or {}

def load_tickets_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Loads tickets from CSV and parses the JSON issue field."""
    tickets = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Issue is a JSON array
                issue_history = json.loads(row["Issue"])
                # Extract text for easier searching later
                full_text = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in issue_history])
                row["parsed_issue"] = issue_history
                row["full_text"] = full_text
            except json.JSONDecodeError:
                row["parsed_issue"] = []
                row["full_text"] = row["issue"]
            tickets.append(row)
    logger.info(f"Loaded {len(tickets)} tickets from {csv_path}")
    return tickets

def chunk_markdown(content: str) -> List[str]:
    import re
    # Split by markdown headers
    chunks = re.split(r'\n(?=#+ )', content)
    # Filter out empty or very small chunks
    valid_chunks = [c.strip() for c in chunks if len(c.strip()) > 20]
    return valid_chunks if valid_chunks else [content.strip()]

def load_corpus() -> List[Document]:
    """Loads all markdown files from the data directory and chunks them."""
    documents = []
    repo_root = DATA_DIR.parent
    
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".md"):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(repo_root).as_posix()
                
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                        chunks = chunk_markdown(content)
                        for i, chunk_text in enumerate(chunks):
                            doc = Document(
                                filepath=rel_path,
                                content=chunk_text,
                                metadata={
                                    "company": full_path.parts[-2] if len(full_path.parts) > 1 else "unknown",
                                    "chunk_id": i
                                }
                            )
                            documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to read {full_path}: {e}")
                    
    logger.info(f"Loaded {len(documents)} document chunks from corpus.")
    return documents
