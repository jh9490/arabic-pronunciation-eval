#!/usr/bin/env python3
"""POC wrapper to launch Microsoft's VibeVoice demos."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def resolve_demo_path(repo_dir: Path, demo: str) -> Path:
    """Locate the requested demo script inside the VibeVoice repo."""
    demo_map = {
        "asr": repo_dir / "demo" / "vibevoice_asr_gradio_demo.py",
        "realtime": repo_dir / "demo" / "vibevoice_realtime_demo.py",
    }
    return demo_map[demo]


def pick_device(demo_args: List[str]) -> Optional[str]:
    """Return a device override if not already specified in demo_args."""
    if "--device" in demo_args:
        return None
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the VibeVoice demos via a local repo clone"
    )
    parser.add_argument(
        "--demo",
        choices=["asr", "realtime"],
        default="asr",
        help="Which demo to run (default: asr)",
    )
    parser.add_argument(
        "--repo-dir",
        required=True,
        help="Path to a local clone of https://github.com/microsoft/VibeVoice",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="HF model id or local model path",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to run the demo (default: current Python)",
    )
    parser.add_argument(
        "demo_args",
        nargs=argparse.REMAINDER,
        help="Extra args for the demo script (prefix with --)",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).expanduser().resolve()
    if not repo_dir.exists():
        raise SystemExit(f"Repo dir not found: {repo_dir}")

    demo_path = resolve_demo_path(repo_dir, args.demo)
    if not demo_path.exists():
        raise SystemExit(
            "Demo script not found at: "
            f"{demo_path}\n"
            "Check the upstream repo for the correct demo path."
        )

    model_path = args.model_path
    if not model_path:
        model_path = (
            "microsoft/VibeVoice-ASR"
            if args.demo == "asr"
            else "microsoft/VibeVoice-Realtime-0.5B"
        )

    cmd = [
        args.python,
        str(demo_path),
        "--model_path",
        model_path,
    ]

    if args.demo == "realtime":
        device = pick_device(args.demo_args)
        if device:
            cmd.extend(["--device", device])

    if args.demo_args:
        cmd.extend(args.demo_args)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
