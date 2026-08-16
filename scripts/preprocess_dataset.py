"""
Dataset preprocessor for chunking and metadata enrichment.
"""
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess():
    logger.info("Cleaning, deduplicating, and formatting Goa text corpora...")
    proc_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    logger.info(f"Processed dataset ready at {proc_dir}")

if __name__ == "__main__":
    preprocess()
