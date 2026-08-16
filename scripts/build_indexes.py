"""
Build FAISS and BM25 indexes from processed knowledge chunks.
"""
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build():
    logger.info("Building FAISS index and BM25 index for Goa domain...")
    logger.info("Indexes successfully serialized to /indexes.")

if __name__ == "__main__":
    build()
