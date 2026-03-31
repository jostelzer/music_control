#!/usr/bin/env python3
"""Headless Magenta player for Music Floor."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
DEFAULT_CONTROL_HOST = "127.0.0.1"
DEFAULT_CONTROL_PORT = 8014

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


class _PlayerSessionBridge:
    def __init__(
        self,
        client,
        *,
        style_text: str,
        style_tokens: list[int],
        style_source: str,
        generation_kwargs: dict[str, object],
        adaptive_playback: bool,
        target_buffer_frames: int,
        max_buffer_frames: int,
    ) -> None:
        self._client = client
        self._lock = threading.RLock()
        self._style_text = str(style_text)
        self._style_tokens = [int(token) for token in style_tokens]
        self._style_source = str(style_source)
        self._generation_kwargs = dict(generation_kwargs)
        self._adaptive_playback = bool(adaptive_playback)
        self._target_buffer_frames = int(target_buffer_frames)
        self._max_buffer_frames = int(max_buffer_frames)

    def _snapshot_locked(self) -> dict[str, object]:
        playback_state = str(getattr(self._client, "playback_state", "unknown"))
        status_text_fn = getattr(self._client, "status_text", None)
        if callable(status_text_fn):
            status_text = str(status_text_fn())
        else:
            status_text = playback_state
        return {
            "status": "ok",
            "server_url": str(getattr(self._client, "server_url", "")),
            "style_text": self._style_text,
            "style_source": self._style_source,
            "style_tokens": list(self._style_tokens),
            "generation_kwargs": dict(self._generation_kwargs),
            "adaptive_playback": self._adaptive_playback,
            "target_buffer_frames": self._target_buffer_frames,
            "max_buffer_frames": self._max_buffer_frames,
            "session_is_active": playback_state != "STOPPED",
            "playback_state": playback_state,
            "status_text": status_text,
        }

    @staticmethod
    def _coerce_style_tokens(raw_tokens) -> list[int]:
        if not isinstance(raw_tokens, (list, tuple)):
            raise ValueError("style_tokens must be a list of integers.")
        return [int(round(float(token))) for token in raw_tokens]

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._snapshot_locked()

    def update_style_tokens(
        self,
        style_tokens,
        *,
        style_text: str | None = None,
        style_source: str = "manual",
    ) -> dict[str, object]:
        tokens = self._coerce_style_tokens(style_tokens)
        text = self._style_text if style_text is None else str(style_text).strip() or self._style_text
        source = str(style_source).strip() or "manual"
        with self._lock:
            returned_tokens = list(
                self._client.update_style_tokens(
                    tokens,
                    style_text=text,
                    style_source=source,
                )
            )
            self._style_text = text
            self._style_source = source
            self._style_tokens = returned_tokens
            return self._snapshot_locked()

    def reset_session(
        self,
        style_tokens,
        *,
        style_text: str | None = None,
        style_source: str = "manual",
    ) -> dict[str, object]:
        tokens = self._coerce_style_tokens(style_tokens)
        text = self._style_text if style_text is None else str(style_text).strip() or self._style_text
        source = str(style_source).strip() or "manual"
        with self._lock:
            self._client.reset_session(
                style_text=text,
                style_tokens=tokens,
                style_source=source,
                generation_kwargs=dict(self._generation_kwargs),
                adaptive_playback=self._adaptive_playback,
                target_buffer_frames=self._target_buffer_frames,
                max_buffer_frames=self._max_buffer_frames,
            )
            self._style_text = text
            self._style_source = source
            current_tokens = getattr(self._client, "_current_style_tokens", tokens)
            self._style_tokens = [int(token) for token in current_tokens]
            return self._snapshot_locked()


def _build_player_control_handler(bridge: _PlayerSessionBridge):
    class _PlayerControlHandler(BaseHTTPRequestHandler):
        server_version = "MusicFloorPlayerControl/1.0"

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return None

        def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object body.")
            return payload

        def do_GET(self) -> None:
            if self.path != "/state":
                self._send_json(404, {"status": "error", "error": "not found"})
                return
            self._send_json(200, bridge.snapshot())

        def do_POST(self) -> None:
            try:
                payload = self._read_json_body()
                if self.path == "/update_style_tokens":
                    response = bridge.update_style_tokens(
                        payload.get("style_tokens"),
                        style_text=payload.get("style_text"),
                        style_source=str(payload.get("style_source") or "manual"),
                    )
                    self._send_json(200, response)
                    return
                if self.path == "/reset_session":
                    response = bridge.reset_session(
                        payload.get("style_tokens"),
                        style_text=payload.get("style_text"),
                        style_source=str(payload.get("style_source") or "manual"),
                    )
                    self._send_json(200, response)
                    return
                if self.path == "/state":
                    self._send_json(200, bridge.snapshot())
                    return
                self._send_json(404, {"status": "error", "error": "not found"})
            except Exception as exc:
                self._send_json(400, {"status": "error", "error": str(exc)})

    return _PlayerControlHandler


class _PlayerControlServer:
    def __init__(self, host: str, port: int, bridge: _PlayerSessionBridge) -> None:
        self._server = ThreadingHTTPServer((host, int(port)), _build_player_control_handler(bridge))
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="music-floor-player-control",
            daemon=True,
        )

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


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
    parser.add_argument(
        "--control-host",
        type=str,
        default=DEFAULT_CONTROL_HOST,
        help="Host for the local floor-control API exposed by this player.",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=DEFAULT_CONTROL_PORT,
        help="Port for the local floor-control API. Use 0 to disable it.",
    )
    return parser.parse_args()


def _seconds_to_frames(seconds: float, frame_length_samples: int, sample_rate: int) -> int:
    seconds = max(0.0, float(seconds))
    frame_seconds = float(frame_length_samples) / float(sample_rate)
    if frame_seconds <= 0:
        raise ValueError("Frame duration must be positive.")
    return int(math.ceil(seconds / frame_seconds))


def _frames_to_seconds(frames: int, frame_length_samples: int, sample_rate: int) -> float:
    frames = max(0, int(frames))
    frame_seconds = float(frame_length_samples) / float(sample_rate)
    if frame_seconds <= 0:
        raise ValueError("Frame duration must be positive.")
    return frames * frame_seconds


def _prompt_was_explicitly_requested(args: argparse.Namespace) -> bool:
    if args.prompt_file or args.music_prompt:
        return True
    if any((prompt or "").strip() for prompt in args.row_prompt):
        return True
    return any(
        token == "--prompt-bank" or token.startswith("--prompt-bank=")
        for token in sys.argv[1:]
    )


def _fetch_existing_style_tokens(client) -> list[int] | None:
    request_json = getattr(client, "_request_json", None)
    if not callable(request_json):
        return None
    response = request_json("POST", "/control", json={})
    tokens = response.get("style_tokens") if isinstance(response, dict) else None
    if not isinstance(tokens, list):
        return None
    try:
        return [int(token) for token in tokens]
    except (TypeError, ValueError):
        return None


def _create_realtime_client(module, args):
    client_cls = module.RealtimeClient
    try:
        parameters = set(inspect.signature(client_cls).parameters)
    except (TypeError, ValueError):
        parameters = {"server_url", "local_audio", "output_device"}

    kwargs = {
        "local_audio": not args.no_local_audio,
        "output_device": args.output_device,
    }
    if "audio_backend" in parameters:
        kwargs["audio_backend"] = args.audio_backend
    elif args.audio_backend:
        os.environ["MAGENTA_RT_AUDIO_BACKEND"] = str(args.audio_backend)
        print(
            "RealtimeClient does not accept --audio-backend directly; using "
            "MAGENTA_RT_AUDIO_BACKEND fallback.",
            file=sys.stderr,
            flush=True,
        )
    if "audio_latency" in parameters:
        kwargs["audio_latency"] = args.audio_latency
    elif args.audio_latency is not None:
        print(
            "RealtimeClient does not support --audio-latency; ignoring.",
            file=sys.stderr,
            flush=True,
        )
    if "audio_blocksize" in parameters:
        kwargs["audio_blocksize"] = args.audio_blocksize
    elif args.audio_blocksize != DEFAULT_AUDIO_BLOCKSIZE:
        print(
            "RealtimeClient does not support --audio-blocksize; ignoring.",
            file=sys.stderr,
            flush=True,
        )
    return client_cls(args.magenta_server_url, **kwargs)


def main() -> int:
    args = parse_args()
    if args.list_prompt_banks:
        print(format_prompt_bank_listing())
        return 0

    shutdown = _ShutdownController(DEFAULT_SHUTDOWN_TIMEOUT_SECONDS)
    shutdown.install()
    client = None
    control_server = None

    try:
        module = _load_magenta_client_module()
        client = _create_realtime_client(module, args)

        prompt_bank_name, prompt_rows = resolve_prompt_rows(
            prompt_bank=args.prompt_bank,
            prompt_file=args.prompt_file,
            row_prompts=args.row_prompt,
            legacy_music_prompt=args.music_prompt,
        )
        client.refresh_stream_info()
        prompt_requested = _prompt_was_explicitly_requested(args)
        prompt = prompt_rows[0].prompt
        startup_style_source = "prompt"
        startup_style_text = prompt
        tokens = None
        if not prompt_requested:
            try:
                existing_tokens = _fetch_existing_style_tokens(client)
            except Exception as exc:
                existing_tokens = None
                print(
                    f"Could not fetch existing server style tokens: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            if existing_tokens:
                tokens = list(existing_tokens)
                startup_style_source = "preserved"
                startup_style_text = "preserved server style"
        if tokens is None:
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
        active_context_frames = int(
            generation_kwargs.get(
                "context_length_frames",
                int(args.context_frames) if args.context_frames is not None else 0,
            )
        )
        active_decode_frames = int(generation_kwargs.get("max_decode_frames", args.decode_window))
        active_crossfade_frames = int(
            round(
                int(generation_kwargs.get("crossfade_samples", 0))
                / float(max(1, client.frame_length_samples))
            )
        )
        default_context_frames_fn = getattr(client, "default_context_frames", None)
        default_context_frames = (
            int(default_context_frames_fn()) if callable(default_context_frames_fn) else None
        )

        client.start_session(
            style_text=startup_style_text,
            style_tokens=tokens,
            style_source="prompt" if startup_style_source == "prompt" else "manual",
            generation_kwargs=generation_kwargs,
            adaptive_playback=args.adaptive_playback,
            target_buffer_frames=target_buffer_frames,
            max_buffer_frames=max_buffer_frames,
        )

        if args.control_port > 0:
            control_bridge = _PlayerSessionBridge(
                client,
                style_text=startup_style_text,
                style_tokens=tokens,
                style_source="prompt" if startup_style_source == "prompt" else "manual",
                generation_kwargs=generation_kwargs,
                adaptive_playback=args.adaptive_playback,
                target_buffer_frames=target_buffer_frames,
                max_buffer_frames=max_buffer_frames,
            )
            control_server = _PlayerControlServer(
                args.control_host,
                args.control_port,
                control_bridge,
            )
            control_server.start()

        print(f"server={client.server_url}", flush=True)
        print(f"prompt_bank={prompt_bank_name}", flush=True)
        print(f"startup_style_source={startup_style_source}", flush=True)
        print(f"prompt={startup_style_text}", flush=True)
        print(f"style_tokens={tokens}", flush=True)
        if control_server is not None:
            print(f"control_url={control_server.url}", flush=True)
        print(
            "generation="
            f"emit_frames={active_decode_frames} "
            f"context_frames={active_context_frames} "
            f"crossfade_frames={active_crossfade_frames} "
            f"temperature={float(generation_kwargs.get('temperature', args.temperature)):.2f} "
            f"topk={int(generation_kwargs.get('topk', args.topk))} "
            f"guidance_weight={float(generation_kwargs.get('guidance_weight', args.guidance_weight)):.2f}",
            flush=True,
        )
        if default_context_frames is not None:
            print(f"context_default_frames={default_context_frames}", flush=True)
        print(
            "buffer_frames="
            f"target={target_buffer_frames} "
            f"max={max_buffer_frames} "
            f"adaptive_playback={bool(args.adaptive_playback)}",
            flush=True,
        )
        print(
            "buffer_seconds="
            f"target={_frames_to_seconds(target_buffer_frames, client.frame_length_samples, client.sample_rate):.3f} "
            f"max={_frames_to_seconds(max_buffer_frames, client.frame_length_samples, client.sample_rate):.3f}",
            flush=True,
        )
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
            if control_server is not None:
                control_server.close()
        finally:
            try:
                if client is not None:
                    client.stop_session()
            finally:
                shutdown.finish()

    return shutdown.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
