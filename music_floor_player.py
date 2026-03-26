#!/usr/bin/env python3
"""Headless Magenta player for Music Floor."""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import sys
import time
from pathlib import Path

DEFAULT_MUSIC_PROMPT = "ambient synths"
DEFAULT_MAGENTA_SERVER_URL = "http://graciosa:8013"
DEFAULT_DECODE_WINDOW = 2
DEFAULT_CROSSFADE_FRAMES = 1
DEFAULT_TEMPERATURE = 1.1
DEFAULT_TOPK = 40
DEFAULT_GUIDANCE_WEIGHT = 5.0
DEFAULT_TARGET_BUFFER_FRAMES = 4
DEFAULT_MAX_BUFFER_FRAMES = 12
DEFAULT_ADAPTIVE_PLAYBACK = False

_MAGENTA_CLIENT_MODULE = None


def _load_magenta_client_module():
    global _MAGENTA_CLIENT_MODULE
    if _MAGENTA_CLIENT_MODULE is not None:
        return _MAGENTA_CLIENT_MODULE

    repo_root = Path(__file__).resolve().parent
    candidates = [
        repo_root.parent / "magenta-realtime" / "demos" / "gradio_client.py",
        Path.home() / "git" / "magenta-realtime" / "demos" / "gradio_client.py",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "magenta_realtime_gradio_client",
            candidate,
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MAGENTA_CLIENT_MODULE = module
        return module

    raise SystemExit(
        "Could not locate the sibling 'magenta-realtime' repo. "
        "Expected it at ../magenta-realtime or ~/git/magenta-realtime."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless Magenta player for Music Floor."
    )
    parser.add_argument(
        "--music-prompt",
        type=str,
        default=DEFAULT_MUSIC_PROMPT,
        help="Text prompt used to seed the initial six style tokens.",
    )
    parser.add_argument(
        "--magenta-server-url",
        type=str,
        default=DEFAULT_MAGENTA_SERVER_URL,
        help="URL of the Magenta realtime server.",
    )
    parser.add_argument(
        "--no-local-audio",
        action="store_true",
        help="Disable Python-side local audio playback.",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="Optional sounddevice output device index.",
    )
    parser.add_argument(
        "--decode-window",
        type=int,
        default=DEFAULT_DECODE_WINDOW,
        help="Magenta decode window in frames.",
    )
    parser.add_argument(
        "--crossfade-frames",
        type=int,
        default=DEFAULT_CROSSFADE_FRAMES,
        help="Crossfade size in codec frames.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOPK,
        help="Top-k sampling cutoff.",
    )
    parser.add_argument(
        "--guidance-weight",
        type=float,
        default=DEFAULT_GUIDANCE_WEIGHT,
        help="Guidance weight.",
    )
    parser.add_argument(
        "--target-buffer-frames",
        type=int,
        default=DEFAULT_TARGET_BUFFER_FRAMES,
        help="Target audio buffer depth in frames.",
    )
    parser.add_argument(
        "--max-buffer-frames",
        type=int,
        default=DEFAULT_MAX_BUFFER_FRAMES,
        help="Maximum audio buffer depth in frames.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=5.0,
        help="Seconds between status prints. Use 0 to disable periodic status.",
    )
    parser.add_argument(
        "--adaptive-playback",
        action="store_true",
        default=DEFAULT_ADAPTIVE_PLAYBACK,
        help="Enable adaptive playback timing. Disabled by default to match the Gradio demo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = _load_magenta_client_module()
    client = module.RealtimeClient(
        args.magenta_server_url,
        local_audio=not args.no_local_audio,
        output_device=args.output_device,
    )
    atexit.register(client.stop_session)

    prompt = (args.music_prompt or "").strip() or DEFAULT_MUSIC_PROMPT
    tokens = list(client.update_style(prompt))
    generation_kwargs = module._build_generation_kwargs(
        client,
        decode_window=args.decode_window,
        crossfade_frames=args.crossfade_frames,
        temperature=args.temperature,
        topk=args.topk,
        guidance_weight=args.guidance_weight,
        fixed_seed_enabled=False,
        fixed_seed=0,
    )

    client.start_session(
        style_text=prompt,
        style_tokens=tokens,
        style_source="prompt",
        generation_kwargs=generation_kwargs,
        adaptive_playback=args.adaptive_playback,
        target_buffer_frames=args.target_buffer_frames,
        max_buffer_frames=args.max_buffer_frames,
    )

    print(f"server={client.server_url}", flush=True)
    print(f"prompt={prompt}", flush=True)
    print(f"style_tokens={tokens}", flush=True)
    if args.status_interval > 0:
        print(client.status_text(), flush=True)

    next_status_time = time.monotonic() + max(0.1, args.status_interval)
    try:
        while True:
            time.sleep(0.1)
            if args.status_interval <= 0:
                continue
            now = time.monotonic()
            if now < next_status_time:
                continue
            print(client.status_text(), flush=True)
            next_status_time = now + max(0.1, args.status_interval)
    except KeyboardInterrupt:
        pass
    finally:
        client.stop_session()


if __name__ == "__main__":
    main()
