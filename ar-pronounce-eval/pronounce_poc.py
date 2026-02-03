#!/usr/bin/env python3
"""Minimal POC for Arabic pronunciation similarity using wav2vec2 hidden states.

Inputs:
  - reference_audio.wav   (perfect pronunciation)
  - spoken_audio.wav      (different speaker)
  - reference_word        (Arabic word, fully diacritized)

Output:
  - Reference word
  - Similarity score
  - Decision: SAME / DIFFERENT / IGNORE
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchaudio
from sklearn.metrics.pairwise import cosine_similarity
from transformers import Wav2Vec2Model, Wav2Vec2Processor


# Similarity thresholds
SAME_THRESHOLD = 0.85
DIFF_THRESHOLD = 0.75


def load_audio(path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """Load audio file and convert to mono with target sample rate."""
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr


def trim_silence(wav: torch.Tensor, sr: int, top_db: float = 30.0) -> torch.Tensor:
    """Trim leading/trailing silence using energy-based VAD."""
    # Prefer sox effects when available; otherwise fall back to simple energy trim.
    if hasattr(torchaudio, "sox_effects"):
        # Apply a small gain to avoid overly aggressive trimming on quiet speech.
        effects = [
            ["gain", "-n"],
            ["silence", "1", "0.1", f"{top_db}d"],
            ["reverse"],
            ["silence", "1", "0.1", f"{top_db}d"],
            ["reverse"],
        ]
        wav = wav.unsqueeze(0)
        trimmed, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)
        return trimmed.squeeze(0)

    # Fallback: trim by RMS energy threshold.
    frame_len = int(0.02 * sr)  # 20ms
    hop_len = int(0.01 * sr)    # 10ms
    if wav.numel() < frame_len:
        return wav
    frames = wav.unfold(0, frame_len, hop_len)
    rms = torch.sqrt(torch.mean(frames**2, dim=1) + 1e-8)
    max_rms = torch.max(rms)
    if max_rms <= 0:
        return wav
    # Convert top_db to linear ratio
    thresh = max_rms / (10 ** (top_db / 20.0))
    idx = torch.nonzero(rms >= thresh, as_tuple=False).squeeze()
    if idx.numel() == 0:
        return wav
    start = int(idx[0].item() * hop_len)
    end = int(min(wav.numel(), (idx[-1].item() * hop_len) + frame_len))
    return wav[start:end]


def normalize_loudness(wav: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
    """Normalize waveform to a target RMS."""
    rms = torch.sqrt(torch.mean(wav**2) + 1e-8)
    if rms > 0:
        wav = wav * (target_rms / rms)
    return wav


def preprocess_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load, trim silence, and normalize audio."""
    wav, sr = load_audio(path, target_sr)
    wav = trim_silence(wav, sr)
    wav = normalize_loudness(wav)
    return wav


def extract_embedding(
    wav: torch.Tensor, sr: int, processor, model, device: torch.device
) -> np.ndarray:
    """Extract speaker-invariant embedding by mean pooling hidden states."""
    inputs = processor(wav.numpy(), sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # (batch, frames, hidden)
        embedding = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
    return embedding


def decide(similarity: float) -> str:
    """Decision logic based on similarity thresholds."""
    if similarity >= SAME_THRESHOLD:
        return "SAME"
    if similarity <= DIFF_THRESHOLD:
        return "DIFFERENT"
    return "IGNORE"


def read_reference_word(path: Path) -> str:
    """Read reference word from a local text file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Reference text file is empty: {path}")
    return text


def record_audio(path: Path, duration: float, sr: int = 16000) -> None:
    """Record audio from microphone and save as WAV."""
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "sounddevice is required for recording. Install with: pip install sounddevice"
        ) from exc

    print(f"Recording {duration:.1f}s -> {path}")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    wav = torch.from_numpy(audio.T).contiguous()
    torchaudio.save(str(path), wav, sr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Arabic pronunciation similarity POC")
    parser.add_argument(
        "--reference-audio",
        default="audio_1.wav",
        help="Path to reference audio WAV (default: audio_1.wav)",
    )
    parser.add_argument(
        "--spoken-audio",
        default="spoken_audio.wav",
        help="Path to spoken_audio.wav (default: spoken_audio.wav)",
    )
    parser.add_argument(
        "--reference-text",
        default="text_1.txt",
        help="Path to reference text file (default: text_1.txt)",
    )
    parser.add_argument(
        "--record",
        choices=["none", "reference", "spoken", "both"],
        default="spoken",
        help="Record audio from mic for reference/spoken/both (default: spoken)",
    )
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ask to record spoken audio before evaluation (default: false)",
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=4.0,
        help="Recording duration in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--model",
        default="facebook/wav2vec2-base",
        help="HuggingFace Wav2Vec2 model (hidden states used)",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    reference_audio = Path(args.reference_audio)
    spoken_audio = Path(args.spoken_audio)
    reference_text = Path(args.reference_text)

    if args.record in {"reference", "both"}:
        record_audio(reference_audio, args.record_duration)
    if args.record in {"spoken", "both"}:
        record_audio(spoken_audio, args.record_duration)
    elif args.interactive:
        prompt = (
            f"Record spoken audio now for {args.record_duration:.1f}s? [Y/n]: "
        )
        answer = input(prompt).strip().lower()
        if answer in {"", "y", "yes"}:
            record_audio(spoken_audio, args.record_duration)

    if not reference_audio.exists():
        print(f"Missing reference audio: {reference_audio}", file=sys.stderr)
        sys.exit(1)
    if not spoken_audio.exists():
        print(f"Missing spoken audio: {spoken_audio}", file=sys.stderr)
        sys.exit(1)
    if not reference_text.exists():
        print(f"Missing reference text: {reference_text}", file=sys.stderr)
        sys.exit(1)

    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2Model.from_pretrained(args.model)
    model.to(device)
    model.eval()

    ref_wav = preprocess_audio(args.reference_audio)
    spk_wav = preprocess_audio(args.spoken_audio)

    ref_emb = extract_embedding(ref_wav, 16000, processor, model, device)
    spk_emb = extract_embedding(spk_wav, 16000, processor, model, device)

    similarity = float(cosine_similarity([ref_emb], [spk_emb])[0][0])
    decision = decide(similarity)

    print(f"Reference word: {read_reference_word(reference_text)}")
    print(f"Similarity score: {similarity:.4f}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
