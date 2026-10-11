#!/usr/bin/env python3
"""
scripts/06_download_indiccorp.py — Download IndicCorpV2 data from HuggingFace.

Downloads the IndicCorpV2 dataset (ai4bharat/IndicCorpV2) for Hindi and Nepali.
This is a "downloaded" data source — a pre-curated corpus built for LM training.

IndicCorpV2 structure on HuggingFace:
  - Hindi files: data/hi-1.txt, data/hi-2.txt, data/hi-3.txt (each ~20–30 GB)
  - Nepali file: data/ne.txt (single file)

By default, downloads only hi-1.txt for Hindi (to conserve disk space).
Use --all-parts to download all three Hindi files.

The data is one sentence per line in plain text format.

Output: <lang>/data/raw_downloaded/indiccorp/

Usage:
  python scripts/06_download_indiccorp.py --lang hindi
  python scripts/06_download_indiccorp.py --lang hindi --all-parts
  python scripts/06_download_indiccorp.py --lang nepali
"""

import argparse
import os
import sys

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.utils import (
    count_tokens, count_lines, get_data_dir, setup_logging,
    update_manifest_source
)

logger = setup_logging(__name__)

# ============================================================
# File mappings for IndicCorpV2
# ============================================================

INDICCORP_FILES = {
    "hindi": {
        "default": ["hi-1.txt"],
        "all": ["hi-1.txt", "hi-2.txt", "hi-3.txt"],
    },
    "nepali": {
        "default": ["ne.txt"],
        "all": ["ne.txt"],
    },
}

REPO_ID = "ai4bharat/IndicCorpV2"


def download_indiccorp_file_streaming(filename: str, output_dir: str, max_tokens: int) -> tuple[str, int, int]:
    """
    Stream a single file from the IndicCorpV2 HuggingFace repo until max_tokens.
    Implements resume and retries for robust downloading of large files.
    """
    import requests
    import time
    
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/data/{filename}"
    local_path = os.path.join(output_dir, filename)
    
    logger.info(f"Streaming {filename} from {url} up to {max_tokens:,} tokens...")
    
    doc_count = 0
    token_count = 0
    
    # Pre-count existing tokens if file exists (useful for resuming across script runs)
    if os.path.exists(local_path):
        logger.info(f"File {filename} exists. Calculating existing tokens to resume...")
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                doc_count += 1
                token_count += count_tokens(line)
        logger.info(f"Resuming from {doc_count:,} docs, ~{token_count:,} tokens.")
        
    if token_count >= max_tokens:
        logger.info(f"Already reached token limit ({max_tokens:,}) for {filename}.")
        return local_path, doc_count, token_count

    max_retries = 10
    retries = 0
    
    while retries < max_retries and token_count < max_tokens:
        headers = {}
        file_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
        if file_size > 0:
            headers["Range"] = f"bytes={file_size}-"
            
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as response:
                # If server doesn't support Range, it will return 200 instead of 206
                if response.status_code == 200 and file_size > 0:
                    logger.warning("Server ignored Range header. Restarting file from beginning.")
                    file_size = 0
                    doc_count = 0
                    token_count = 0
                    mode = "wb"
                elif response.status_code == 416: # Range not satisfiable (e.g. file is complete)
                    logger.info("File already fully downloaded according to server.")
                    break
                else:
                    response.raise_for_status()
                    mode = "ab" if file_size > 0 else "wb"
                    
                with open(local_path, mode) as f:
                    with tqdm(desc=f"Tokens ({filename})", total=max_tokens, initial=token_count, unit="tok") as pbar:
                        for line_bytes in response.iter_lines():
                            if not line_bytes:
                                continue
                            
                            f.write(line_bytes + b"\n")
                            
                            # We can decode and count tokens line by line
                            line = line_bytes.decode("utf-8", errors="ignore")
                            doc_count += 1
                            tokens_in_line = count_tokens(line)
                            token_count += tokens_in_line
                            pbar.update(tokens_in_line)
                            
                            if token_count >= max_tokens:
                                logger.info(f"Reached token limit ({max_tokens:,}) for {filename}.")
                                break
            
            # If we exit the context manager successfully, download is complete or limit reached
            break
            
        except Exception as e:
            retries += 1
            logger.warning(f"Connection error while streaming {filename}: {e}. Retrying ({retries}/{max_retries}) in 5 seconds...")
            time.sleep(5)
            
    if retries >= max_retries:
        logger.error(f"Failed to download {filename} after {max_retries} retries.")
        
    logger.info(f"Saved to: {local_path} ({doc_count:,} docs, ~{token_count:,} tokens)")
    return local_path, doc_count, token_count


def count_file_stats(filepath: str, sample_lines: int = 10000) -> dict:
    """
    Count lines and estimate token count for a large text file.

    For very large files, we sample the first N lines and extrapolate
    the token count rather than reading the entire file.

    Args:
        filepath: Path to the text file.
        sample_lines: Number of lines to sample for estimation.

    Returns:
        Dict with 'line_count' and 'estimated_tokens'.
    """
    line_count = 0
    sample_token_count = 0
    sampled = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line_count += 1
            if i < sample_lines:
                sample_token_count += count_tokens(line)
                sampled += 1

    # Extrapolate token count from sample
    if sampled > 0:
        avg_tokens_per_line = sample_token_count / sampled
        estimated_tokens = int(avg_tokens_per_line * line_count)
    else:
        estimated_tokens = 0

    return {
        "line_count": line_count,
        "estimated_tokens": estimated_tokens,
        "sample_size": sampled,
    }


def download_indiccorp(lang: str, all_parts: bool = False, max_tokens: int = 500_000_000) -> None:
    """
    Download IndicCorpV2 data for one language.

    Args:
        lang: Language name ('hindi' or 'nepali').
        all_parts: If True, download all parts for Hindi (3 files).
        max_tokens: Maximum number of tokens to download per language.
    """
    if lang not in INDICCORP_FILES:
        logger.error(f"No IndicCorpV2 config for language: {lang}")
        return

    output_dir = str(get_data_dir(lang) / "raw_downloaded" / "indiccorp")
    os.makedirs(output_dir, exist_ok=True)

    key = "all" if all_parts else "default"
    files_to_download = INDICCORP_FILES[lang][key]

    total_docs = 0
    total_tokens = 0

    for filename in files_to_download:
        if total_tokens >= max_tokens:
            logger.info(f"Already reached total token limit ({max_tokens:,}). Skipping {filename}.")
            break

        remaining_tokens = max_tokens - total_tokens
        filepath, doc_count, token_count = download_indiccorp_file_streaming(
            filename, output_dir, remaining_tokens
        )
        
        total_docs += doc_count
        total_tokens += token_count

    # Update manifest
    update_manifest_source(
        lang=lang,
        source_name="indiccorp",
        category="downloaded",
        doc_count=total_docs,
        token_count=total_tokens,
        extra={
            "files": files_to_download,
            "repo_id": REPO_ID,
        },
    )

    logger.info(
        f"IndicCorpV2 {lang}: {total_docs:,} total docs, "
        f"~{total_tokens:,} tokens"
    )


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Download IndicCorpV2 data from HuggingFace."
    )
    parser.add_argument(
        "--lang",
        choices=["hindi", "nepali", "all"],
        required=True,
        help="Language to download, or 'all' for both."
    )
    parser.add_argument(
        "--all-parts",
        action="store_true",
        default=False,
        help="Download all parts for Hindi (hi-1, hi-2, hi-3). Default: hi-1 only."
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500_000_000,
        help="Maximum tokens to download per language (default: 500M)."
    )

    args = parser.parse_args()

    if args.lang == "all":
        for lang in ["hindi", "nepali"]:
            download_indiccorp(lang, args.all_parts, args.max_tokens)
    else:
        download_indiccorp(args.lang, args.all_parts, args.max_tokens)


if __name__ == "__main__":
    main()
