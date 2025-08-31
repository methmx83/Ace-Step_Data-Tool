# THIRD_PARTY_LICENSES.md

This file lists third-party code used in ACE-DATA_v2, the license under which it is provided, and which files in this repository include or are derived from that code. Where possible replace the placeholder SHAs with the exact commit hashes from the upstream repositories to improve traceability.

1) ACE-Step
   - Repo: https://github.com/ace-step/ACE-Step
   - License: Apache-2.0
  - Upstream commit (reference): 6ae0852b1388de6dc0cca26b31a86d711f723cb3  # repo HEAD at time of snapshot (recommend replacing with file-level commit SHA)
   - Files included (unchanged):
     - scripts/train/acestep/language_segmentation/LangSegment.py
     - scripts/train/acestep/language_segmentation/language_filters.py
     - scripts/train/acestep/language_segmentation/utils/num.py
     - scripts/train/acestep/models/ace_step_transformer.py
     - scripts/train/acestep/models/attention.py
     - scripts/train/acestep/models/config.json
     - scripts/train/acestep/models/customer_attention_processor.py
     - scripts/train/acestep/models/lyrics_utils/lyric_encoder.py
     - scripts/train/acestep/models/lyrics_utils/lyric_normalizer.py
     - scripts/train/acestep/models/lyrics_utils/lyric_tokenizer.py
     - scripts/train/acestep/models/lyrics_utils/zh_num2words.py
     - scripts/train/acestep/models/lyrics_utils/vocab.json
     - scripts/train/acestep/music_dcae/music_dcae_pipeline.py
     - scripts/train/acestep/music_dcae/music_log_mel.py
     - scripts/train/acestep/music_dcae/music_vocoder.py
     - scripts/train/acestep/schedulers/scheduling_flow_match_euler_discrete.py
     - scripts/train/acestep/schedulers/scheduling_flow_match_heun_discrete.py
     - scripts/train/acestep/schedulers/scheduling_flow_match_pingpong.py
     - scripts/train/acestep/apg_guidance.py
     - scripts/train/acestep/cpu_offload.py
     - scripts/train/acestep/text2music_dataset.py
     - scripts/train/acestep/data_sampler.py
     - scripts/train/acestep/pipeline_ace_step.py

2) woct0rdho/ACE-Step (fork)
   - Repo: https://github.com/woct0rdho/ACE-Step
   - License: Apache-2.0
  - Upstream commit (reference): ea2ec6ba68be9c73254c2ec8f89a6965c6e5c3e8  # repo HEAD at time of snapshot (recommend replacing with file-level commit SHA)
   - Files copied and modified in this repository:
     - scripts/train/convert2hf_dataset_new.py (modified)
     - scripts/train/preprocess_dataset_new.py (modified)
     - scripts/train/trainer_new.py -> scripts/train/trainer_optimized.py (modified and renamed)
   - Notes: These files were adapted to the ACE-DATA_v2 project. Typical modifications introduced by the current maintainer include:
     - Windows path normalization (backslash-safe paths)
     - HDF5/tensor storage and access adjustments
     - Torch optimisations and quantization hints for reduced VRAM
     - Renaming and reorganisation to match ACE-DATA_v2 layout
   - Required actions for compliance: retain original copyright/license headers in any modified files (do not remove existing headers). Add a header comment in files you modified that documents the original source, a short change summary and the Apache-2.0 license reference.

General notes:
- The Apache-2.0 license requires that the license text is distributed with the work. The repository root contains a `LICENSE` file with the Apache-2.0 text.
- Retain original copyright and license headers in any copied files. Do not remove or alter these headers.
- Replace the placeholder SHAs with exact upstream commit hashes when possible. This improves traceability for security and audit purposes.

How to obtain upstream commit SHAs (recommended):

PowerShell / cmd steps (example):

```powershell
git clone https://github.com/ace-step/ACE-Step temp-ace
cd temp-ace
git log -n1 --pretty=format:%H -- scripts/train/acestep
```

Or use the GitHub web UI: open the file in the upstream repo, click "History" and copy the commit SHA for the version you used.

Suggested file-header template for modified files (add at top of each modified file):

```text
# Original: <upstream path> from <upstream repo URL>
# Upstream commit: <REPLACE_WITH_COMMIT_SHA>
# Modifications: (brief list of changes applied, e.g. "Windows path normalization; VRAM optimizations; HDF5 changes")
# License: Apache-2.0 (see repository LICENSE)
```

If you want, I can try to locate the likely upstream commit SHAs for the included files and propose replacements — say if you want me to attempt that.
