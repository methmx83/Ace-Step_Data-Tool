"""
Shared logging utilities for the lyrics scraper.

The original ACE‑Step Data‑Tool used a simple logging module to
accumulate messages in a global list and echo them to the console.
Keeping this module separate allows code that depends on it to
operate unchanged.  The log messages are stored in the ``LOGS``
list and also printed to the terminal when ``log_message`` is
called.

The ``LOGS`` list can be inspected by external code (for example,
in a Gradio UI) to display a live log to the user.
"""

from __future__ import annotations

# A global list used to collect log messages.  External code may
# read from this list to display logs.
LOGS: list[str] = []

def log_message(message: str) -> None:
    """Append a message to the log and print it to stdout.

    Parameters
    ----------
    message: str
        The message to record.  A blank line will automatically
        follow the message in both the log and the console output
        to improve readability.
    """
    global LOGS
    LOGS.append(message)
    LOGS.append("")  # Append a blank line to separate messages
    # Print to the terminal as well.  In a Gradio UI these prints
    # will appear in the server logs rather than the user interface.
    print(message)
    print("")
