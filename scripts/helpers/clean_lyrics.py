"""
Post‑processing of lyric files.

This module implements the cleaning logic used in the ACE‑Step
Data‑Tool.  After lyrics are scraped and saved to disk the files
often begin with metadata lines such as ``Artist: ...`` and
``Title: ...``.  The :func:`bereinige_datei` function removes
everything above the first line that begins with a ``[`` (which
typically marks a section such as ``[Verse 1]`` or ``[Chorus]``).
It then writes the cleaned lyrics back to the file.

The :func:`main` function can be executed from the command line to
recursively clean all ``*_lyrics.txt`` files under the project’s
``data`` directory.  In the standalone Gradio UI these functions
will be called programmatically rather than via ``main``.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable

from .shared_logs import LOGS, log_message


def bereinige_datei(dateipfad: str) -> None:
    """Clean a single lyrics file in place.

    This function searches for the first line beginning with ``[`` and
    removes all lines prior to it.  If no such line is found the file
    remains unchanged.

    Parameters
    ----------
    dateipfad: str
        Path to the lyrics file that should be cleaned.
    """
    try:
        log_message(f"⏳ Processing lyrics: {dateipfad}")
        with open(dateipfad, 'r', encoding='utf-8') as datei:
            zeilen = datei.readlines()

        # Find the index of the first line that starts with '['
        start_index = None
        for i, zeile in enumerate(zeilen):
            if zeile.strip().startswith('['):
                start_index = i
                break

        # If such a line is found, write the part starting at this index
        # Otherwise, skip the file (remain unchanged)
        if start_index is not None:
            bereinigte_zeilen = zeilen[start_index:]
            # Overwrite the original file with the cleaned content
            with open(dateipfad, 'w', encoding='utf-8') as datei:
                datei.writelines(bereinigte_zeilen)
            log_message(f"💾 Lyrics cleaned up: {dateipfad}")
        else:
            # Inform that no section marker was found; file left unchanged.
            log_message(f"⚠️ Skipped (no '[' line found): {dateipfad}")
    except Exception as e:
        log_message(f"❌ Error processing {dateipfad}: {e}")


def main() -> None:
    """Search recursively for ``*_lyrics.txt`` files and clean them.

    This function is provided for convenience when running this
    module as a script.  In the context of the Gradio UI the
    cleaning will be invoked directly on the files created by the
    scraper, so calling :func:`main` is unnecessary.
    """
    # Determine the base directory relative to this file; default to
    # ../data to match the original project structure
    start_ordner = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
    log_message(f"🔍 Search folders for existing lyrics: {start_ordner}")
    # Recursively search for all *_lyrics.txt files
    suchmuster = os.path.join(start_ordner, '**', '*_lyrics.txt')
    lyrics_dateien: Iterable[str] = glob.glob(suchmuster, recursive=True)
    if not lyrics_dateien:
        log_message("⚠️ No files found with the name '*_lyrics.txt'.")
        return
    log_message(f"📂 Found files: {len(list(lyrics_dateien))}")
    for dateipfad in lyrics_dateien:
        bereinige_datei(dateipfad)
    log_message("✅ Processing completed.")


if __name__ == '__main__':
    main()
