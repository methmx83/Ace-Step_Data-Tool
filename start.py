from pathlib import Path
import subprocess
import sys
import os


def ensure_data_dirs_runtime():
    """Erzeuge die benötigten DATA_DIRS.
    """
    DATA_DIRS = [
        Path("data"),
        Path("data") / "audio",
        Path("data") / "cache",
        Path("data") / "lora",
        Path("data") / "data_sets" / "jsons_sets",
        Path("data") / "data_sets" / "train_set",
    ]

    created = []

    # 1) Versuch: im aktuellen Arbeitsverzeichnis
    for rel in DATA_DIRS:
        try:
            target = Path.cwd() / rel
            target.mkdir(parents=True, exist_ok=True)
            created.append(target)
        except Exception:
            # Fehler beim Schreiben ins CWD ignorieren und später Fallback versuchen
            continue

    if created:
        for p in created:
            print(f"Ensured data dir at: {p}")
        return

    # 2) Fallback: nur wenn das Paketverzeichnis nach Repo-Root aussieht
    try:
        pkg_root = Path(__file__).resolve().parent
        def looks_like_repo(d: Path) -> bool:
            return (d / "setup.py").exists() or (d / ".git").exists()

        if looks_like_repo(pkg_root):
            created_pkg = []
            for rel in DATA_DIRS:
                try:
                    target = pkg_root / rel
                    target.mkdir(parents=True, exist_ok=True)
                    created_pkg.append(target)
                except Exception:
                    continue
            if created_pkg:
                for p in created_pkg:
                    print(f"Ensured data dir at: {p}")
                return
            else:
                print("Warning: could not create data dirs under package root (permissions?)")
        else:
            print("Warning: package path does not look like a repository root; skipping package-relative dir creation to avoid writing into site-packages")
    except Exception as exc:
        print(f"Warning: could not create fallback data dirs: {exc}")


def main():
    ensure_data_dirs_runtime()
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