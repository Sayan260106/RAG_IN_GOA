"""
Dataset downloader script for Goa Knowledge Base (heritage, cuisine, places, culture).
"""
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing Goa dataset download...")
    target_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Raw dataset initialized at {target_dir}")

if __name__ == "__main__":
    main()
