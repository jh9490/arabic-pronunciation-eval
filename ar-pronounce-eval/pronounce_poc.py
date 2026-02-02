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
    # torchaudio.functional.vad removes leading silence only; use sox effect for both
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Arabic pronunciation similarity POC")
    parser.add_argument("--reference-audio", required=True, help="Path to reference_audio.wav")
    parser.add_argument("--spoken-audio", required=True, help="Path to spoken_audio.wav")
    parser.add_argument("--reference-word", required=True, help="Arabic word (fully diacritized)")
    parser.add_argument(
        "--model",
        default="facebook/wav2vec2-base",
        help="HuggingFace Wav2Vec2 model (hidden states used)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print(f"Reference word: {args.reference_word}")
    print(f"Similarity score: {similarity:.4f}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
