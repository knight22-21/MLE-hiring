import csv
import sys
import time
import os

# Ensure the parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.ingestion import load_tickets_from_csv, load_corpus
from pipeline.retrieval import HybridRetriever
from pipeline.generation import generate_ticket_response
from config import OUTPUT_CSV, INPUT_CSV, SAMPLE_CSV, logger

def main(use_sample=False):
    start_time = time.time()
    
    # 1. Load Data
    csv_to_process = SAMPLE_CSV if use_sample else INPUT_CSV
    logger.info(f"Processing tickets from: {csv_to_process}")
    
    tickets = load_tickets_from_csv(csv_to_process)
    corpus = load_corpus()
    
    # 2. Initialize Retriever
    retriever = HybridRetriever(corpus)
    
    # 3. Process Tickets
    results = []
    for i, ticket in enumerate(tickets):
        logger.info(f"Processing ticket {i+1}/{len(tickets)}")
        
        # Simple query extraction (just using the full text for BM25)
        query = f"{ticket.get('subject', '')} {ticket.get('full_text', '')}"
        
        retrieved_docs = retriever.search(query, top_k=3)
        
        output_obj = generate_ticket_response(ticket, retrieved_docs)
        csv_dict = output_obj.to_csv_dict(
            issue=ticket["Issue"], 
            subject=ticket["Subject"], 
            company=ticket["Company"]
        )
        results.append(csv_dict)
        
    # 4. Write Output
    if results:
        headers = list(results[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Successfully wrote {len(results)} outputs to {OUTPUT_CSV}")
    
    end_time = time.time()
    logger.info(f"Execution completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # pass --sample flag to run on the sample dataset
    use_sample = "--sample" in sys.argv
    main(use_sample)
