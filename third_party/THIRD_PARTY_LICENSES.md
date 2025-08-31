# THIRD_PARTY_LICENSES.md

This file lists third-party code used in ACE-DATA_v2, the license under which it is provided, and which files in this repository include or are derived from that code. Replace the placeholder SHAs with the exact commit hashes when available.

1) ACE-Step
   - Repo: https://github.com/ace-step/ACE-Step
   - License: Apache-2.0
   - Upstream commit (reference): <ACE_STEP_COMMIT_SHA>
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
   - Upstream commit (reference): <WOCT0RDHO_COMMIT_SHA>
   - Files copied and modified in this repository:
     - scripts/train/convert2hf_dataset_new.py (modified)
     - scripts/train/preprocess_dataset_new.py (modified)
     - scripts/train/trainer_new.py -> scripts/train/trainer_optimized.py (modified and renamed)
   - Notes: These files were adapted to the ACE-DATA_v2 project (Windows path normalization, HDF5 and torch optimisations, quantization hints). The original license headers are retained where present, and additional modification lines were added by the current maintainer.

General notes:
- The Apache-2.0 license requires that the license text is distributed with the work. Ensure `LICENSE` (Apache-2.0) is present in the repository root.
- Retain original copyright and license headers in any copied files.
- If exact upstream commit hashes are known, replace the placeholders above. This improves traceability for security and audit purposes.

If you need help finding the exact upstream commit SHAs, run:

```powershell
git clone https://github.com/ace-step/ACE-Step temp-ace
cd temp-ace
git log -n1 --pretty=format:%H -- scripts/train/acestep
```

Or use the GitHub web UI to inspect file history and copy the commit SHA from the commit page.
