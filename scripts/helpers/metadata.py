"""
Helpers for normalising and cleaning metadata strings.

These functions originate from the ACE‑Step Data‑Tool project and
have been preserved for compatibility.  They are responsible for
sanitising track names and artist names when constructing URLs to
fetch lyrics from Genius.com or when creating file names on disk.

The functions include:

* :func:`clean_filename` – strip illegal filesystem characters from a string.
* :func:`clean_rap_metadata` – remove common annotations such as track numbers,
  bracketed descriptors and "feat." segments.
* :func:`normalize_feature_artists` – replace different variants of "feat."
  with "and" when constructing display strings.
* :func:`normalize_string` – convert a string to an ASCII hyphenated form
  suitable for URL paths.
"""

from __future__ import annotations

import re
import unicodedata

def clean_filename(name: str) -> str:
    """Remove characters that are illegal in file names.

    Parameters
    ----------
    name: str
        The original file name.

    Returns
    -------
    str
        A version of the name with illegal characters removed.
    """
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


def clean_rap_metadata(text: str) -> str:
    """Strip common adornments from track titles and artists.

    This helper removes leading track numbers, any content enclosed in
    round or square brackets, and trailing "feat."/"ft." annotations.

    Parameters
    ----------
    text: str
        The string to clean, such as a track title or artist name.

    Returns
    -------
    str
        The cleaned string.
    """
    # Remove leading track numbers like "01. "
    text = re.sub(r'^\d+\.\s*', '', text)
    # Remove any parenthesised or bracketed phrases
    text = re.sub(r'\(.*?\)|\[.*?\]', '', text)
    # Remove feat/ft indications and anything following them, case insensitive
    text = re.sub(r'(?i)\s*(feat\..*|ft\..*|featuring.*)', '', text)
    return text.strip()


def normalize_feature_artists(artist: str) -> str:
    """Convert various feat indicators to 'and'.

    When constructing user facing strings (not necessarily URLs), it can
    be useful to normalise "feat." segments into a more readable
    conjunction.  This helper replaces the variant spellings of
    "feat." with " and ".

    Parameters
    ----------
    artist: str
        The original artist string.

    Returns
    -------
    str
        A version of the artist string with feat. indicators normalised.
    """
    return re.sub(r'(?i)\s*feat\.?\s*', ' and ', artist)


def normalize_string(text: str) -> str:
    """Normalise a string to ASCII and hyphenate it for URL paths.

    The function performs several steps:

    * Replaces a number of Unicode dash and hyphen characters with the
      ASCII hyphen ``-``.
    * Uses NFKD normalization and transliteration to remove accents
      and diacritics.
    * Strips any remaining characters that are not alphanumeric,
      whitespace or hyphen.
    * Converts whitespace runs to single hyphens and trims leading
      or trailing hyphens.

    Parameters
    ----------
    text: str
        The string to normalise.

    Returns
    -------
    str
        The normalised and hyphenated string.
    """
    # Replace various unicode dash characters with a simple hyphen
    dash_chars = [
        '\u2010',  # Hyphen
        '\u2011',  # Non‑breaking hyphen
        '\u2012',  # Figure dash
        '\u2013',  # En dash
        '\u2014',  # Em dash
        '\u2015',  # Horizontal bar
        '\u2212',  # Minus sign
        '\u2043',  # Hyphen bullet
        '\uFE58',  # Small em dash
        '\uFE63',  # Small hyphen‑minus
        '\uFF0D',  # Fullwidth hyphen‑minus
    ]
    for dash in dash_chars:
        text = text.replace(dash, '-')
    # Decompose accented characters and drop diacritics
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    # Remove any character that is not a word character, whitespace or hyphen
    text = re.sub(r'[^\w\s-]', '', text.lower())
    # Collapse runs of whitespace into single hyphens and strip leading/trailing hyphens
    return re.sub(r'\s+', '-', text).strip('-')
