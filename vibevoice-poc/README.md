# VibeVoice POC (Microsoft)

Small, local proof-of-concept for running the VibeVoice **ASR** or **Realtime TTS** demos from Microsoft's VibeVoice repo.

## Why this POC
- The official VibeVoice repo has removed the **TTS** code (as of 2025-09-05), but the ASR and Realtime releases remain documented.
- This POC supports **ASR** (`--demo asr`) and **Realtime TTS** (`--demo realtime`).

## Prerequisites
- Python 3.10+ (recommended)
- Git
- A GPU is helpful but not required for a quick demo.

## Setup (manual)
1) Clone the Microsoft VibeVoice repo (outside this folder).
   - Repo: https://github.com/microsoft/VibeVoice

2) Create and activate a venv, then install the repo's requirements.

3) Download the model from Hugging Face:
   - ASR: https://huggingface.co/microsoft/VibeVoice-ASR
   - Realtime: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B

4) Optional: install the repo in editable mode so the demo script can import it.

## Run the demo via this POC wrapper
From this folder:

```bash
python run_realtime_demo.py \
  --repo-dir /path/to/VibeVoice \
  --demo asr \
  --model-path microsoft/VibeVoice-ASR
```

You can pass extra args (e.g., `--port`, `--host`, or any demo-specific flags) after `--`:

```bash
python run_realtime_demo.py \
  --repo-dir /path/to/VibeVoice \
  --demo asr \
  --model-path microsoft/VibeVoice-ASR \
  -- --host 0.0.0.0 --port 7860
```

Realtime TTS:

```bash
python run_realtime_demo.py \
  --repo-dir /path/to/VibeVoice \
  --demo realtime \
  --model-path microsoft/VibeVoice-Realtime-0.5B
```

## Notes
- If the demo script path changes in the upstream repo, update it in `run_realtime_demo.py`.
- If Hugging Face prompts for access, accept the model terms in your HF account first.
