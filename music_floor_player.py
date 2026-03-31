#!/usr/bin/env python3
"""Headless Magenta player for Music Floor."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

from floor_slots.music_floor_prompt_banks import (
    DEFAULT_PROMPT_BANK,
    format_prompt_bank_listing,
    resolve_prompt_rows,
)

DEFAULT_MAGENTA_SERVER_URL = "http://graciosa:8013"
DEFAULT_DECODE_WINDOW = 2
DEFAULT_CROSSFADE_FRAMES = 1
DEFAULT_TEMPERATURE = 1.1
DEFAULT_TOPK = 40
DEFAULT_GUIDANCE_WEIGHT = 5.0
DEFAULT_TARGET_BUFFER_FRAMES = 4
DEFAULT_MAX_BUFFER_FRAMES = 12
DEFAULT_ADAPTIVE_PLAYBACK = False
DEFAULT_AUDIO_BLOCKSIZE = 0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 3.0

_MAGENTA_CLIENT_MODULE = None


class _ShutdownController:
    def __init__(self, timeout_seconds: float):
        self._timeout_seconds = max(0.0, float(timeout_seconds))
        self._stop_requested = threading.Event()
        self._force_exit_cancel: threading.Event | None = None
        self._exit_code = 0
        self._previous_handlers: dict[int, signal.Handlers] = {}

    @property
    def exit_code(self) -> int:
        return self._exit_code

    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    def install(self) -> None:
        for signum in (
            getattr(signal, "SIGINT", None),
            getattr(signal, "SIGTERM", None),
        ):
            if signum is None:
                continue
            try:
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
            except (OSError, RuntimeError, ValueError):
                continue

    def finish(self) -> None:
        self._cancel_force_exit()
        for signum, handler in self._previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, RuntimeError, ValueError):
                continue
        self._previous_handlers.clear()

    def request_stop(self, exit_code: int, reason: str) -> None:
        if self._stop_requested.is_set():
            print(f"{reason} Forcing exit.", file=sys.stderr, flush=True)
            os._exit(exit_code)
        self._exit_code = exit_code
        self._stop_requested.set()
        print(reason, file=sys.stderr, flush=True)
        self._arm_force_exit(exit_code)

    def _handle_signal(self, signum: int, _frame) -> None:
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = f"signal {signum}"
        self.request_stop(128 + int(signum), f"{signal_name} received. Stopping player...")

    def _arm_force_exit(self, exit_code: int) -> None:
        if self._timeout_seconds <= 0 or self._force_exit_cancel is not None:
            return
        cancel_event = threading.Event()
        self._force_exit_cancel = cancel_event

        def _force_exit() -> None:
            if cancel_event.wait(timeout=self._timeout_seconds):
                return
            print(
                "Shutdown did not finish in "
                f"{self._timeout_seconds:.1f}s. Forcing exit.",
                file=sys.stderr,
                flush=True,
            )
            os._exit(exit_code)

        threading.Thread(
            target=_force_exit,
            name="music-floor-force-exit",
            daemon=True,
        ).start()

    def _cancel_force_exit(self) -> None:
        if self._force_exit_cancel is None:
            return
        self._force_exit_cancel.set()
        self._force_exit_cancel = None


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
        "--prompt-bank",
        type=str,
        default=DEFAULT_PROMPT_BANK,
        help="Built-in six-row prompt bank for startup style prompts.",
    )
    parser.add_argument(
        "--row-prompt",
        action="append",
        default=[],
        help="Override a row prompt. Repeat up to six times; overrides rows from the top.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Optional text or JSON file defining exactly six prompt rows.",
    )
    parser.add_argument(
        "--list-prompt-banks",
        action="store_true",
        help="Print the built-in prompt banks and exit.",
    )
    parser.add_argument(
        "--music-prompt",
        type=str,
        default=None,
        help="Legacy alias for overriding row 1 only.",
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
        "--emit-frames",
        dest="decode_window",
        type=int,
        default=DEFAULT_DECODE_WINDOW,
        help="Emit frames per step / max decode frames.",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=None,
        help="Override context length in codec frames. Defaults to the server-provided value.",
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
        "--target-buffer-seconds",
        type=float,
        default=None,
        help="Target audio buffer depth in seconds. Overrides --target-buffer-frames.",
    )
    parser.add_argument(
        "--max-buffer-frames",
        type=int,
        default=DEFAULT_MAX_BUFFER_FRAMES,
        help="Maximum audio buffer depth in frames.",
    )
    parser.add_argument(
        "--max-buffer-seconds",
        type=float,
        default=None,
        help="Maximum audio buffer depth in seconds. Overrides --max-buffer-frames.",
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
    parser.add_argument(
        "--audio-backend",
        choices=("callback", "blocking", "virtual", "blocking_virtual"),
        default=None,
        help="Local audio backend. Use 'blocking' for a more buffered path on flaky wireless links.",
    )
    parser.add_argument(
        "--audio-latency",
        type=str,
        default=None,
        help="sounddevice latency setting for local audio playback. Accepts 'low', 'high', or a numeric seconds value.",
    )
    parser.add_argument(
        "--audio-blocksize",
        type=int,
        default=DEFAULT_AUDIO_BLOCKSIZE,
        help="sounddevice blocksize for local audio playback. Use 0 to let the backend choose.",
    )
    return parser.parse_args()


def _seconds_to_frames(seconds: float, frame_length_samples: int, sample_rate: int) -> int:
    seconds = max(0.0, float(seconds))
    frame_seconds = float(frame_length_samples) / float(sample_rate)
    if frame_seconds <= 0:
        raise ValueError("Frame duration must be positive.")
    return int(math.ceil(seconds / frame_seconds))


def main() -> int:
    args = parse_args()
    if args.list_prompt_banks:
        print(format_prompt_bank_listing())
        return 0

    shutdown = _ShutdownController(DEFAULT_SHUTDOWN_TIMEOUT_SECONDS)
    shutdown.install()
    client = None

    try:
        module = _load_magenta_client_module()
        client = module.RealtimeClient(
            args.magenta_server_url,
            local_audio=not args.no_local_audio,
            output_device=args.output_device,
            audio_backend=args.audio_backend,
            audio_latency=args.audio_latency,
            audio_blocksize=args.audio_blocksize,
        )

        prompt_bank_name, prompt_rows = resolve_prompt_rows(
            prompt_bank=args.prompt_bank,
            prompt_file=args.prompt_file,
            row_prompts=args.row_prompt,
            legacy_music_prompt=args.music_prompt,
        )
        client.refresh_stream_info()
        prompt = prompt_rows[0].prompt
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
        if args.context_frames is not None:
            generation_kwargs["context_length_frames"] = int(args.context_frames)
        target_buffer_frames = int(args.target_buffer_frames)
        max_buffer_frames = int(args.max_buffer_frames)
        if args.target_buffer_seconds is not None:
            target_buffer_frames = _seconds_to_frames(
                args.target_buffer_seconds,
                client.frame_length_samples,
                client.sample_rate,
            )
        if args.max_buffer_seconds is not None:
            max_buffer_frames = _seconds_to_frames(
                args.max_buffer_seconds,
                client.frame_length_samples,
                client.sample_rate,
            )

        client.start_session(
            style_text=prompt,
            style_tokens=tokens,
            style_source="prompt",
            generation_kwargs=generation_kwargs,
            adaptive_playback=args.adaptive_playback,
            target_buffer_frames=target_buffer_frames,
            max_buffer_frames=max_buffer_frames,
        )

        print(f"server={client.server_url}", flush=True)
        print(f"prompt_bank={prompt_bank_name}", flush=True)
        print(f"prompt={prompt}", flush=True)
        print(f"style_tokens={tokens}", flush=True)
        if args.status_interval > 0:
            print(client.status_text(), flush=True)

        next_status_time = time.monotonic() + max(0.1, args.status_interval)
        while not shutdown.stop_requested():
            time.sleep(0.1)
            if args.status_interval <= 0:
                continue
            now = time.monotonic()
            if now < next_status_time:
                continue
            print(client.status_text(), flush=True)
            next_status_time = now + max(0.1, args.status_interval)
    except KeyboardInterrupt:
        shutdown.request_stop(130, "Keyboard interrupt received. Stopping player...")
    finally:
        try:
            if client is not None:
                client.stop_session()
        finally:
            shutdown.finish()

    return shutdown.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
