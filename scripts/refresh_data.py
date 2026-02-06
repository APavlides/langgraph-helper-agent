#!/usr/bin/env python3
"""
Script to download/refresh the llms.txt documentation files.

This script downloads the latest versions of the LangGraph and LangChain
documentation in llms.txt format. It's designed to be run manually or
as part of a CI/CD pipeline for scheduled updates.

Usage:
    python scripts/refresh_data.py
    python scripts/refresh_data.py --full  # Include full versions
"""

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import httpx

# Documentation sources
SOURCES = {
    "langgraph": {
        "short": "https://langchain-ai.github.io/langgraph/llms.txt",
        "full": "https://langchain-ai.github.io/langgraph/llms-full.txt",
    },
    "langchain": {
        "short": "https://docs.langchain.com/llms.txt",
        "full": "https://docs.langchain.com/llms-full.txt",
    },
}

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data"


def compute_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def download_file(url: str, timeout: float = 30.0) -> str | None:
    """
    Download a file from URL.

    Args:
        url: URL to download
        timeout: Request timeout in seconds

    Returns:
        File content as string, or None if download failed
    """
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPError as e:
        print(f"  ❌ Failed to download {url}: {e}")
        return None


def save_file(content: str, filepath: Path) -> bool:
    """
    Save content to file, checking if it changed.

    Args:
        content: Content to save
        filepath: Path to save to

    Returns:
        True if file was updated, False if unchanged
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Check if content changed
    if filepath.exists():
        existing_content = filepath.read_text()
        if compute_hash(existing_content) == compute_hash(content):
            return False

    filepath.write_text(content)
    return True


def refresh_data(include_full: bool = False, verbose: bool = True) -> dict:
    """
    Download all documentation files.

    Args:
        include_full: Whether to download full versions (larger files)
        verbose: Whether to print progress

    Returns:
        Dictionary with download results
    """
    results = {
        "downloaded": [],
        "unchanged": [],
        "failed": [],
        "timestamp": datetime.now(UTC).isoformat(),
    }

    if verbose:
        print("📥 Refreshing documentation data...")
        print(f"   Output directory: {DATA_DIR}")
        print()

    for source_name, urls in SOURCES.items():
        # Always download short version
        versions_to_download = ["short"]
        if include_full:
            versions_to_download.append("full")

        for version in versions_to_download:
            url = urls[version]
            filename = f"{source_name}_llms{'_full' if version == 'full' else ''}.txt"
            filepath = DATA_DIR / filename

            if verbose:
                print(f"📄 {source_name} ({version})...")

            content = download_file(url)

            if content is None:
                results["failed"].append(filename)
                continue

            updated = save_file(content, filepath)

            if updated:
                results["downloaded"].append(filename)
                if verbose:
                    print(f"   ✅ Updated: {filename} ({len(content):,} bytes)")
            else:
                results["unchanged"].append(filename)
                if verbose:
                    print(f"   ⏭️  Unchanged: {filename}")

    # Save metadata
    metadata_path = DATA_DIR / "metadata.txt"
    metadata = f"""# LangGraph Helper Agent - Documentation Data
# Last updated: {results['timestamp']}
# Files: {', '.join(results['downloaded'] + results['unchanged'])}
"""
    metadata_path.write_text(metadata)

    if verbose:
        print()
        print("📊 Summary:")
        print(f"   Updated: {len(results['downloaded'])}")
        print(f"   Unchanged: {len(results['unchanged'])}")
        print(f"   Failed: {len(results['failed'])}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download/refresh llms.txt documentation files"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include full versions (larger files, more comprehensive)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    results = refresh_data(
        include_full=args.full,
        verbose=not args.quiet,
    )

    # Exit with error if any downloads failed
    if results["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
