"""
Helper module for detecting the tempo (BPM) of audio files.

This module exposes a single function, ``detect_tempo``, which takes the path to
an audio file and returns its estimated tempo in beats per minute (BPM).  It
uses ``librosa`` under the hood to perform onset detection and tempo
estimation.  The implementation is adapted from ``get_bpm.py`` but wrapped in
a function name and signature that matches the expectations of ``tagger.py``.

If tempo estimation fails for any reason, the function will return ``None``
instead of raising an exception.  This allows callers to gracefully handle
tracks without a detectable tempo.

Example:

    >>> from scripts.helpers.bpm import detect_tempo
    >>> bpm = detect_tempo("/path/to/song.mp3")
    >>> if bpm:
    ...     print(f"Detected {bpm} BPM")
    ... else:
    ...     print("No tempo detected")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# ``numpy`` and ``librosa`` are optional dependencies.  They are imported
# lazily within the functions that require them.  This avoids raising an
# ImportError at module import time when those packages are not installed,
# allowing the caller to decide how to handle missing dependencies.


def _dealias_halftime_double(bpm: float, lo: float, hi: float) -> float:
    """Ensure BPM stays within the specified range by halving or doubling."""
    while bpm < lo:
        bpm *= 2
    while bpm > hi:
        bpm /= 2
    return bpm


def _snap_off_by_one(bpm: float, thresh: float) -> float:
    """
    Snap the BPM to the nearest integer if it is within a specified threshold.

    This is helpful because tempo estimation can occasionally return values that
    are close to whole numbers but slightly off due to noise in the input
    signal.  By snapping values that are, for example, within 0.6 BPM of an
    integer, the result becomes more predictable.
    """
    r = round(bpm)
    return float(r) if abs(bpm - r) <= thresh else bpm


def detect_bpm_librosa(
    audio_path: str,
    sr: int = 44100,
    start_bpm: float = 92.0,
    std_bpm: float = 1.0,
    max_tempo: float = 220.0,
    range_lo: float = 70.0,
    range_hi: float = 180.0,
    snap_thresh: float = 0.6,
) -> int:
    """
    Estimate the tempo (in BPM) of an audio file using librosa.

    Parameters
    ----------
    audio_path:
        Path to the audio file to analyse.
    sr:
        Target sampling rate.  The audio will be resampled on load if needed.
    start_bpm:
        Prior estimate of the tempo.  Helps guide the estimation algorithm.
    std_bpm:
        Standard deviation around the prior BPM.  A smaller value constrains
        the estimation to remain closer to ``start_bpm``.
    max_tempo:
        The maximum tempo allowed for initial estimation.  After initial
        estimation the tempo is folded into the ``range_lo``–``range_hi``
        interval.
    range_lo, range_hi:
        Desired output range for the BPM.  If the raw estimate falls outside
        this range it will be doubled or halved until it fits.
    snap_thresh:
        Threshold for snapping near-integer BPM values to the nearest integer.

    Returns
    -------
    int
        The estimated tempo in beats per minute.
    """
    # Import dependencies here to allow the module to be imported even if
    # librosa or numpy is missing.  If the import fails, propagate the
    # exception to the caller so they can handle it.
    import numpy as np  # type: ignore
    import librosa  # type: ignore

    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # Apply harmonic–percussive source separation to emphasise the percussive
    # component, which tends to carry the most salient rhythmic information.
    _, y_p = librosa.effects.hpss(y)
    y_use = y_p

    # Compute the onset envelope with a relatively small hop length to improve
    # temporal resolution.  Using the median as the aggregation function
    # provides a more robust onset strength.
    hop = 256
    oenv = librosa.onset.onset_strength(
        y=y_use, sr=sr, hop_length=hop, aggregate=np.median
    )

    # Estimate the tempo using the onset envelope.  A prior (start_bpm) and
    # its standard deviation (std_bpm) are provided to stabilise the estimate.
    tempi = librosa.feature.tempo(
        onset_envelope=oenv,
        sr=sr,
        hop_length=hop,
        start_bpm=start_bpm,
        std_bpm=std_bpm,
        max_tempo=max_tempo,
    )
    bpm = float(tempi.squeeze())

    # Fold the estimate into the desired range and snap near integers.
    bpm = _dealias_halftime_double(bpm, range_lo, range_hi)
    bpm = _snap_off_by_one(bpm, snap_thresh)
    return int(round(bpm))


def detect_tempo(audio_path: str, **kwargs) -> Optional[int]:
    """
    Wrapper around ``detect_bpm_librosa`` with a simplified signature.

    This function is provided for compatibility with existing code in
    ``tagger.py``, which expects a ``detect_tempo`` function taking only the
    path to the audio file.  Additional keyword arguments may be supplied to
    override the defaults of ``detect_bpm_librosa``.  Any unexpected
    exceptions are caught and will result in ``None`` being returned.

    Parameters
    ----------
    audio_path:
        Path to the audio file to analyse.
    **kwargs:
        Optional keyword arguments forwarded to ``detect_bpm_librosa``.  These
        allow callers to customise aspects such as ``start_bpm`` or
        ``range_lo`` without modifying the function signature.

    Returns
    -------
    Optional[int]
        The estimated tempo in BPM, or ``None`` if estimation fails.
    """
    try:
        # Delegate to the underlying implementation.  Use defaults where
        # arguments are not provided by the caller.
        bpm = detect_bpm_librosa(audio_path, **kwargs)
        # Ensure the result is an integer and not zero; return None if zero.
        return int(bpm) if bpm > 0 else None
    except Exception:
        # Swallow any exceptions (e.g. unsupported file formats, IO errors) and
        # return None so that callers can decide how to proceed.
        return None


# The module can also be executed directly for quick testing.  It accepts an
# audio file and an optional JSON configuration.  When run as a script it will
# print the detected BPM or return 0 on failure.  This entry point mirrors the
# behaviour of the original ``get_bpm.py`` script while encapsulating its
# functionality within this helper module.
if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description=(
            "Detect the tempo of an audio file.  Optionally provide a JSON "
            "configuration file with a top-level 'bpm' object to override "
            "defaults."
        )
    )
    ap.add_argument("audio", help="Path to the audio file")
    ap.add_argument(
        "--cfg",
        default=None,
        help="Optional path to a JSON config file containing BPM parameters",
    )
    args = ap.parse_args()

    # Load custom parameters from config if provided.
    bpm_kwargs = {}
    if args.cfg and Path(args.cfg).exists():
        try:
            cfg = json.loads(Path(args.cfg).read_text(encoding="utf-8"))
            bpm_kwargs = cfg.get("bpm", {})
        except Exception:
            bpm_kwargs = {}

    # Perform tempo detection and print the result.  Fall back to printing 0
    # when detection fails to mirror the behaviour of the original script.
    bpm = detect_tempo(args.audio, **bpm_kwargs)
    print(bpm or 0)