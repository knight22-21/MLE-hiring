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

def load_corpus() -> List[Document]:
    """Loads all markdown files from the data directory."""
    documents = []
    # Base directory to compute relative paths as expected by evaluation
    repo_root = DATA_DIR.parent
    
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".md"):
                full_path = Path(root) / file
                # relative path with forward slashes as expected by the challenge
                rel_path = full_path.relative_to(repo_root).as_posix()
                
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                        # Basic chunking: For simplicity and since docs aren't huge, 
                        # we'll treat the whole file as one chunk for now, 
                        # but in Phase 3 we can improve it.
                        doc = Document(
                            filepath=rel_path,
                            content=content,
                            metadata={"company": full_path.parts[-2] if len(full_path.parts) > 1 else "unknown"}
                        )
                        documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to read {full_path}: {e}")
                    
    logger.info(f"Loaded {len(documents)} markdown documents from corpus.")
    return documents
