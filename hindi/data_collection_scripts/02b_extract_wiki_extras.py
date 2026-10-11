#!/usr/bin/env python3
"""
scripts/02b_extract_wiki_extras.py — Extract clean text from additional Wikimedia dumps.

Uses WikiExtractor to parse Wiktionary, Wikiquote, and Wikibooks dumps.
Similar to 02_extract_wiki.py, but handles multiple sources per language.

Usage:
  python scripts/02b_extract_wiki_extras.py --lang hindi
  python scripts/02b_extract_wiki_extras.py --lang nepali
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.utils import (
    count_tokens, get_data_dir, setup_logging, update_manifest_source
)
import importlib.util
spec = importlib.util.spec_from_file_location("extract_wiki", os.path.join(os.path.dirname(__file__), "02_extract_wiki.py"))
extract_wiki = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_wiki)
extract_with_wikiextractor = extract_wiki.extract_with_wikiextractor
collect_extracted_articles = extract_wiki.collect_extracted_articles

logger = setup_logging(__name__)

# Must match DUMPS keys from 01b
SOURCES = ["wiktionary", "wikiquote", "wikibooks"]

def find_dump_file(data_dir: Path, source_name: str) -> str:
    """Find the downloaded xml.bz2 dump for a given source."""
    source_dir = data_dir / "raw_manual" / source_name
    if not source_dir.exists():
        return None
        
    for filename in os.listdir(source_dir):
        if filename.endswith(".xml.bz2"):
            return str(source_dir / filename)
    return None

def extract_source(lang: str, source_name: str, data_dir: Path) -> None:
    dump_path = find_dump_file(data_dir, source_name)
    if not dump_path:
        logger.debug(f"[{lang}] No {source_name} dump found. Skipping.")
        return

    source_dir = data_dir / "raw_manual" / source_name
    extracted_dir = str(source_dir / "wikiextractor_output")
    output_path = str(source_dir / "extracted.jsonl")

    if os.path.exists(output_path):
        from shared.utils import count_lines
        existing_count = count_lines(output_path)
        logger.info(f"[{lang}] {source_name} output exists ({existing_count} articles). Skipping.")
        return

    logger.info(f"[{lang}] Extracting {source_name}...")
    try:
        extract_with_wikiextractor(dump_path, extracted_dir)
    except Exception as e:
        logger.error(f"[{lang}] Failed to extract {source_name}: {e}")
        return

    articles = collect_extracted_articles(extracted_dir)
    logger.info(f"[{lang}] Collected {len(articles)} {source_name} articles")

    # Save as JSONL and count tokens
    total_tokens = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for article in articles:
            # Override source tag
            article["source"] = source_name
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
            total_tokens += count_tokens(article["text"])

    logger.info(f"[{lang}] Saved {len(articles)} {source_name} articles (~{total_tokens:,} tokens)")

    # Update manifest
    update_manifest_source(
        lang=lang,
        source_name=source_name,
        category="manual",
        doc_count=len(articles),
        token_count=total_tokens,
        extra={"dump_file": os.path.basename(dump_path)},
    )

    # Cleanup intermediate output
    if os.path.exists(extracted_dir):
        shutil.rmtree(extracted_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Extract extra Wikimedia dumps (Wiktionary, Wikiquote, Wikibooks)."
    )
    parser.add_argument(
        "--lang",
        choices=["hindi", "nepali", "all"],
        required=True,
    )

    args = parser.parse_args()
    langs = ["hindi", "nepali"] if args.lang == "all" else [args.lang]

    for lang in langs:
        data_dir = get_data_dir(lang)
        for source in SOURCES:
            extract_source(lang, source, data_dir)

if __name__ == "__main__":
    main()
