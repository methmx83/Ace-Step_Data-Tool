#!/usr/bin/env python3
# startet das Skript scripts/ui/ui.py aus dem Projekt-Hauptverzeichnis
from pathlib import Path
import subprocess
import sys

def main():
    # Projekt-Root ist das Verzeichnis, in dem diese Datei liegt
    root = Path(__file__).resolve().parent
    module = "scripts.ui.ui"
    cmd = [sys.executable, "-m", module] + sys.argv[1:]
    try:
        subprocess.run(cmd, cwd=str(root), check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Fehler: Prozess endete mit Code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
    except KeyboardInterrupt:
        print("Abgebrochen.", file=sys.stderr)
        sys.exit(130)

if __name__ == "__main__":
    main()