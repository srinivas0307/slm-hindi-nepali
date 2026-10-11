#!/usr/bin/env python3
"""
scripts/03_extract_wikisource.py — Extract text from Wikisource XML dumps.

Wikisource contains full-text books and literature — a different kind of
manual data source from Wikipedia (encyclopedic articles). This script
parses the Wikisource dump using mwparserfromhell to convert MediaWiki
markup into plain text.

Currently applicable to:
  - Hindi Wikisource: hiwikisource-latest-pages-articles.xml.bz2
  - Nepali Wikisource: May not exist — script logs and skips gracefully

Filtering:
  - Skips disambiguation pages (titles containing "बहुविकल्पी")
  - Skips author/contributor pages (namespace != 0)
  - Skips very short pages (<100 characters, likely stubs or metadata)

Output: <lang>/data/raw_manual/wikisource/extracted.jsonl

Usage:
  python scripts/03_extract_wikisource.py --lang hindi
  python scripts/03_extract_wikisource.py --lang nepali
"""

import argparse
import bz2
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.utils import (
    count_tokens, get_data_dir, setup_logging,
    update_manifest_source
)

logger = setup_logging(__name__)

# Wikisource dump filenames per language
WIKISOURCE_FILES = {
    "hindi": "hiwikisource-latest-pages-articles.xml.bz2",
    "nepali": "newikisource-latest-pages-articles.xml.bz2",
}

# MediaWiki XML namespace (used in tag matching)
MW_NS = "{http://www.mediawiki.org/xml/export-0.10/}"


def find_wikisource_dump(lang: str) -> str:
    """
    Find the Wikisource dump file for the specified language.

    Args:
        lang: Language name.

    Returns:
        Path to the dump file.

    Raises:
        FileNotFoundError: If no dump file is found.
    """
    ws_dir = get_data_dir(lang) / "raw_manual" / "wikisource"

    # Look for any .xml.bz2 file
    if ws_dir.exists():
        for filename in os.listdir(ws_dir):
            if filename.endswith(".xml.bz2"):
                return str(ws_dir / filename)

    raise FileNotFoundError(
        f"No Wikisource dump found in {ws_dir}. "
        f"This may be expected for {lang} — Wikisource may not exist."
    )


def parse_wikitext_to_plain(wikitext: str) -> str:
    """
    Convert MediaWiki markup to plain text using mwparserfromhell.

    mwparserfromhell is a parser for MediaWiki markup that can extract
    plain text by stripping templates, HTML tags, headings, etc.

    Args:
        wikitext: Raw MediaWiki markup text.

    Returns:
        Cleaned plain text.
    """
    try:
        import mwparserfromhell
        parsed = mwparserfromhell.parse(wikitext)

        # strip_code() removes templates, tags, headings, etc.
        text = parsed.strip_code()

        # Additional cleanup
        # Remove remaining [[Category:...]] links
        text = re.sub(r"\[\[(?:Category|श्रेणी|वर्ग):.*?\]\]", "", text, flags=re.IGNORECASE)

        # Remove file/image links
        text = re.sub(r"\[\[(?:File|Image|चित्र):.*?\]\]", "", text, flags=re.IGNORECASE)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text

    except Exception as e:
        logger.warning(f"mwparserfromhell failed: {e}, falling back to regex cleanup")
        # Fallback: basic regex cleanup
        text = re.sub(r"\{\{[^}]*\}\}", "", wikitext)  # Remove templates
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", wikitext)  # Links
        text = re.sub(r"<[^>]+>", "", text)  # HTML tags
        text = re.sub(r"={2,6}\s*(.*?)\s*={2,6}", r"\1", text)  # Headings
        return text.strip()


def iterate_wiki_dump(dump_path: str):
    """
    Iterate over pages in a MediaWiki XML dump (bz2 compressed).

    Uses iterative XML parsing to handle large dump files without
    loading the entire file into memory.

    Args:
        dump_path: Path to the .xml.bz2 dump file.

    Yields:
        Tuples of (page_id, title, namespace, wikitext).
    """
    logger.info(f"Parsing XML dump: {dump_path}")

    with bz2.open(dump_path, "rt", encoding="utf-8") as f:
        # Use iterparse for memory-efficient XML processing
        context = ET.iterparse(f, events=("end",))

        page_id = None
        title = None
        ns = None
        text = None

        for event, elem in context:
            tag = elem.tag.replace(MW_NS, "").replace(
                "{http://www.mediawiki.org/xml/export-0.11/}", ""
            )

            if tag == "id" and page_id is None:
                # First <id> inside <page> is the page ID
                page_id = elem.text
            elif tag == "title":
                title = elem.text
            elif tag == "ns":
                ns = elem.text
            elif tag == "text":
                text = elem.text or ""
            elif tag == "page":
                # End of a page element — yield the collected data
                if title and text:
                    yield page_id, title, ns, text

                # Reset for next page
                page_id = None
                title = None
                ns = None
                text = None

                # Clear the element to free memory
                elem.clear()


def should_skip_page(title: str, ns: str, text: str) -> bool:
    """
    Determine if a Wikisource page should be skipped.

    Skips:
      - Non-main and non-page namespaces (ns not in "0", "250")
      - Disambiguation pages
      - Very short pages (<100 chars after stripping)
      - Redirect pages

    Args:
        title: Page title.
        ns: MediaWiki namespace (0 = main content, 250 = Page transcribed text).
        text: Raw wikitext content.

    Returns:
        True if the page should be skipped.
    """
    # Only process main namespace (0) and Page namespace (250)
    # NS 250 contains the actual raw text of transcribed books!
    if ns not in ("0", "250"):
        return True

    # Skip redirects
    if text.strip().lower().startswith("#redirect") or text.strip().startswith("#पुनर्निर्देशन"):
        return True

    # Skip disambiguation pages (Hindi)
    if "बहुविकल्पी" in title:
        return True

    # Skip very short pages
    if len(text.strip()) < 100:
        return True

    return False


def extract_wikisource(lang: str) -> None:
    """
    Full Wikisource extraction pipeline for one language.

    Args:
        lang: Language name ('hindi' or 'nepali').
    """
    ws_dir = get_data_dir(lang) / "raw_manual" / "wikisource"
    output_path = str(ws_dir / "extracted.jsonl")

    # Check if already extracted
    if os.path.exists(output_path):
        logger.info(f"Already extracted: {output_path}. Skipping.")
        return

    try:
        dump_path = find_wikisource_dump(lang)
    except FileNotFoundError:
        logger.warning(
            f"No Wikisource dump found for {lang}. "
            f"This is expected if {lang} Wikisource doesn't exist or is too small. "
            f"Skipping Wikisource extraction."
        )
        return

    articles = []
    skipped = 0

    for page_id, title, ns, wikitext in iterate_wiki_dump(dump_path):
        if should_skip_page(title, ns, wikitext):
            skipped += 1
            continue

        # Convert wikitext to plain text
        plain_text = parse_wikitext_to_plain(wikitext)

        if len(plain_text) < 100:
            skipped += 1
            continue

        articles.append({
            "id": page_id,
            "title": title,
            "text": plain_text,
            "source": "wikisource",
        })

    if not articles:
        logger.warning(f"No articles extracted from {lang} Wikisource dump.")
        return

    # Save as JSONL
    total_tokens = 0
    os.makedirs(ws_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
            total_tokens += count_tokens(article["text"])

    logger.info(
        f"Extracted {len(articles)} pages from {lang} Wikisource "
        f"(skipped {skipped}), ~{total_tokens:,} tokens"
    )

    # Update manifest
    update_manifest_source(
        lang=lang,
        source_name="wikisource",
        category="manual",
        doc_count=len(articles),
        token_count=total_tokens,
    )


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Extract text from Wikisource XML dumps."
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
            extract_wikisource(lang)
    else:
        extract_wikisource(args.lang)


if __name__ == "__main__":
    main()
