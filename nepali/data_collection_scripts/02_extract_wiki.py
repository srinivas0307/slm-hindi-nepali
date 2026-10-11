#!/usr/bin/env python3
"""
scripts/02_extract_wiki.py — Extract clean text from Wikipedia XML dumps.

Uses the WikiExtractor library (as a Python module, not a pre-cleaned dataset)
to parse Wikipedia XML dumps into clean article text. This counts as "manual"
data collection because we are doing the extraction and curation ourselves.

Output format: JSONL (one JSON object per line)
  {"id": "12345", "title": "Article Title", "text": "Clean article text..."}

Saves to: <lang>/data/raw_manual/wikipedia/extracted.jsonl

Usage:
  python scripts/02_extract_wiki.py --lang hindi
  python scripts/02_extract_wiki.py --lang nepali
"""

import argparse
import bz2
import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.utils import (
    count_tokens, get_data_dir, setup_logging,
    update_manifest_source
)

logger = setup_logging(__name__)

# Map language name to Wikipedia dump filename prefix
WIKI_PREFIXES = {
    "hindi": "hiwiki",
    "nepali": "newiki",
}


def find_dump_file(lang: str) -> str:
    """
    Find the Wikipedia XML dump file for the specified language.

    Looks in <lang>/data/raw_manual/wikipedia/ for the .xml.bz2 file.

    Args:
        lang: Language name.

    Returns:
        Path to the dump file.

    Raises:
        FileNotFoundError: If no dump file is found.
    """
    wiki_dir = get_data_dir(lang) / "raw_manual" / "wikipedia"
    prefix = WIKI_PREFIXES.get(lang, "")

    # Look for the dump file
    for filename in os.listdir(wiki_dir) if wiki_dir.exists() else []:
        if filename.endswith(".xml.bz2") and prefix in filename:
            return str(wiki_dir / filename)

    raise FileNotFoundError(
        f"No Wikipedia dump found in {wiki_dir}. "
        f"Run scripts/01_download_wiki_dumps.py --lang {lang} first."
    )


def extract_with_wikiextractor(dump_path: str, output_dir: str) -> str:
    """
    Run WikiExtractor on the dump file to produce clean text.

    WikiExtractor is used as a command-line tool (installed via pip) that
    parses the MediaWiki XML format and strips markup, tables, infoboxes,
    and category links to produce clean article text.

    Args:
        dump_path: Path to the .xml.bz2 dump file.
        output_dir: Directory to write WikiExtractor's output.

    Returns:
        Path to the output directory containing extracted files.
    """
    logger.info(f"Running WikiExtractor on {dump_path}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run WikiExtractor
    # --json: output in JSON format (one article per line)
    # --no-templates: don't expand templates (faster, less noise)
    # --processes: use multiple processes for speed
    cmd = [
        sys.executable, "-m", "wikiextractor.WikiExtractor",
        dump_path,
        "--json",
        "--no-templates",
        "--output", output_dir,
        "--compress",
        "--quiet",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("WikiExtractor completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"WikiExtractor failed: {e.stderr}")
        raise

    return output_dir


def collect_extracted_articles(extracted_dir: str) -> list:
    """
    Collect all articles from WikiExtractor's output directory.

    WikiExtractor creates a tree of directories (AA/, AB/, etc.) with
    files named wiki_00, wiki_01, etc. Each file contains multiple
    JSON articles, one per line.

    Args:
        extracted_dir: Path to WikiExtractor's output directory.

    Returns:
        List of article dicts with 'id', 'title', 'text' fields.
    """
    articles = []

    for root, dirs, files in os.walk(extracted_dir):
        for filename in sorted(files):
            filepath = os.path.join(root, filename)
            try:
                # WikiExtractor may produce gzipped files
                if filepath.endswith(".bz2"):
                    import bz2
                    opener = bz2.open
                else:
                    opener = open

                with opener(filepath, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            article = json.loads(line)
                            # WikiExtractor output has: id, url, title, text
                            text = article.get("text", "").strip()

                            # Skip very short articles (stubs, redirects)
                            if len(text) < 100:
                                continue

                            articles.append({
                                "id": article.get("id", ""),
                                "title": article.get("title", ""),
                                "text": text,
                                "source": "wikipedia",
                            })
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")
                continue

    return articles


def extract_wikipedia(lang: str) -> None:
    """
    Full Wikipedia extraction pipeline for one language.

    Steps:
      1. Find the downloaded dump file
      2. Run WikiExtractor on it
      3. Collect all extracted articles
      4. Save as JSONL
      5. Update manifest with statistics

    Args:
        lang: Language name ('hindi' or 'nepali').
    """
    try:
        dump_path = find_dump_file(lang)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    wiki_dir = get_data_dir(lang) / "raw_manual" / "wikipedia"
    extracted_dir = str(wiki_dir / "wikiextractor_output")
    output_path = str(wiki_dir / "extracted.jsonl")

    # Check if already extracted
    if os.path.exists(output_path):
        from shared.utils import count_lines
        existing_count = count_lines(output_path)
        logger.info(
            f"Extraction output already exists: {output_path} "
            f"({existing_count} articles). Skipping. "
            f"Delete the file to re-extract."
        )
        return

    # Step 1: Run WikiExtractor
    extract_with_wikiextractor(dump_path, extracted_dir)

    # Step 2: Collect articles
    logger.info("Collecting extracted articles...")
    articles = collect_extracted_articles(extracted_dir)
    logger.info(f"Collected {len(articles)} articles")

    # Step 3: Save as JSONL
    total_tokens = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
            total_tokens += count_tokens(article["text"])

    logger.info(
        f"Saved {len(articles)} articles to {output_path} "
        f"(~{total_tokens:,} whitespace tokens)"
    )

    # Step 4: Update manifest
    update_manifest_source(
        lang=lang,
        source_name="wikipedia",
        category="manual",
        doc_count=len(articles),
        token_count=total_tokens,
        extra={"dump_file": os.path.basename(dump_path)},
    )

    # Cleanup WikiExtractor's intermediate output to save disk space
    import shutil
    if os.path.exists(extracted_dir):
        shutil.rmtree(extracted_dir)
        logger.info(f"Cleaned up intermediate files: {extracted_dir}")


def main():
    """Entry point: parse arguments and extract Wikipedia articles."""
    parser = argparse.ArgumentParser(
        description="Extract clean text from Wikipedia XML dumps."
    )
    parser.add_argument(
        "--lang",
        choices=["hindi", "nepali", "all"],
        required=True,
        help="Language to extract, or 'all' for both."
    )

    args = parser.parse_args()

    if args.lang == "all":
        for lang in ["hindi", "nepali"]:
            extract_wikipedia(lang)
    else:
        extract_wikipedia(args.lang)


if __name__ == "__main__":
    main()
