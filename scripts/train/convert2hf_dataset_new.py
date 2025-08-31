#!/usr/bin/env python3

# Original: scripts/train/convert2hf_dataset_new.py from ace-step and woct0rdho fork
# Upstream repos:
#   - ACE-Step: https://github.com/ace-step/ACE-Step (repo HEAD: 6ae0852b1388de6dc0cca26b31a86d711f723cb3)
#   - woct0rdho/ACE-Step: https://github.com/woct0rdho/ACE-Step (repo HEAD: ea2ec6ba68be9c73254c2ec8f89a6965c6e5c3e8)
# Modified for: ACE-DATA_v2
# Modified by: methmx83 (ACE-DATA_v2)
# Modification date: 2025-08-31
# Modifications: Windows path normalization, defaults and minor refactors to integrate with ACE-DATA_v2. See `third_party/THIRD_PARTY_LICENSES.md` for provenance and file-level notes.

import argparse
import os

from datasets import Dataset


def create_dataset(data_dir, output_name, default_tag):
    # Formats supported by torchaudio
    extensions = {
        ".aac",
        ".flac",
        ".mp3",
        ".ogg",
        ".wav",
    }

    all_examples = []
    for file in sorted(os.listdir(data_dir)):
        stem, ext = os.path.splitext(file)
        if ext.lower() not in extensions:
            continue

        file_path = os.path.join(data_dir, file)
        stem_path = os.path.join(data_dir, stem)
        prompt_path = stem_path + "_prompt.txt"
        lyric_path = stem_path + "_lyrics.txt"

        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            prompt = default_tag

        if os.path.exists(lyric_path):
            with open(lyric_path, "r", encoding="utf-8") as f:
                lyrics = f.read().strip()
        else:
            lyrics = "[instrumental]"

        tags = [x.strip() for x in prompt.split(",")]
        example = {
            "keys": stem,
            "filename": file_path,
            "tags": tags,
            "speaker_emb_path": "",
            "norm_lyrics": lyrics,
            "recaption": {},
        }
        all_examples.append(example)

    # repeat specified times
    ds = Dataset.from_list(all_examples)
    ds.save_to_disk(output_name)


def main():
    parser = argparse.ArgumentParser(description="Create a dataset from audio files.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"data\data_files",
        help="Directory containing the audio files.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=r"data\data_sets\jsons_sets",
        help="Name of the output dataset.",
    )
    parser.add_argument(
        "--default_tag",
        type=str,
        default="",
        help="Default tag when there is no prompt file.",
    )
    args = parser.parse_args()

    create_dataset(
        data_dir=args.data_dir,
        output_name=args.output_name,
        default_tag=args.default_tag,
    )


if __name__ == "__main__":
    main()
