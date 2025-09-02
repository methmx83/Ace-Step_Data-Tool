import os
from setuptools import setup, find_packages
from setuptools.command.install import install

# zentrale Liste aller relativen Verzeichnisse, die beim Install/Start erzeugt werden sollen
DATA_DIRS = [
    os.path.join("data"),
    os.path.join("data", "audio"),
    os.path.join("data", "cache"),
    os.path.join("data", "lora"),
    os.path.join("data", "data_sets", "jsons_sets"),
    os.path.join("data", "data_sets", "train_set"),
]


def ensure_data_dirs(root=None):
    """Erzeuge alle Pfade in DATA_DIRS relativ zu root (default: Projekt-Root).

    Gibt die Liste der tatsächlich erzeugten absoluten Pfade zurück.
    """
    root = root or os.path.dirname(__file__)
    created = []
    for rel in DATA_DIRS:
        target = os.path.join(root, rel)
        try:
            os.makedirs(target, exist_ok=True)
            created.append(target)
        except Exception:
            # best-effort: Fehler nicht die Installation abbrechen lassen
            continue
    return created


def main():
    ensure_data_dirs()


class PostInstall(install):
    def run(self):
        # idiomatischer Aufruf des Eltern-Install-Verhaltens
        super().run()
        # best-effort: Versuche, die DATA_DIRS im Repo-Root anzulegen
        try:
            repo_root = os.path.dirname(__file__)
            created = ensure_data_dirs(root=repo_root)
            if created:
                print("Created folders (if missing):")
                for p in created:
                    print("  ", p)
            else:
                print("No folders created (they may already exist).")
        except Exception:
            print("Warning: could not create data dirs during install (continuing)")

setup(
    name="Ace-Step_Data-Tool",
    version="1.0",
    author="methmx83",
    description="Ace-Step Dataset Generator",
    packages=find_packages(),  # Automatically find packages in the current directory
    py_modules=["start"],
    install_requires=[
            "accelerate>=1.0.0",
            "gradio==5.39.0",
            "librosa==0.11.0",
            "soundfile==0.13.1",
            "numpy==2.2.6",
            "requests==2.32.4",
            "datasets>=2.20",
            "transformers==4.55.1",
            "triton-windows==3.3.1.post19",
            "bitsandbytes==0.46.1",
            "beautifulsoup4==4.13.4",
            "tinytag==2.1.1",
            "pytorch_lightning==2.5.1",
            "loguru==0.7.3",
            "num2words==0.5.14",
            "spacy==3.8.4",
            "peft>=0.11",
            "prodigyopt>=1.0",
            "py3langid==0.3.0",
            "pypinyin==0.53.0",
            "pandas==2.3.1",
            "tqdm==4.67.1",
            "nltk==3.9.1",
            "matplotlib==3.10.5",
            "huggingface-hub==0.34.3",
            "tokenizers==0.21.4",
            "h5py==3.13.0",
            "hdf5plugin==5.1.0",
            "diffusers>=0.33.0",
            "natsort==8.4.0",
            "rapidfuzz>=3.0",
            "uvicorn>=0.29",
            "tensorboard",
            "tensorboardX",
    ],
    entry_points={
        "console_scripts": [
            "acedata=start:main",
        ],
    },
    python_requires=">=3.11",
    cmdclass={"install": PostInstall},
)
